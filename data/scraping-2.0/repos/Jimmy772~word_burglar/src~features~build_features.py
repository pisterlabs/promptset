from dotenv import load_dotenv
import os

from tqdm import tqdm

import pandas as pd
import psycopg2

from contextlib import closing
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter


def main():    
    with closing(psycopg2.connect(**PARAMS)) as conn:
        with closing(conn.cursor()) as cursor:
            # create table to save embedded books
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS added_book_ids (
                            book_id INTEGER PRIMARY KEY
                            );''')
            # populate the table with existing `book_id`s
            cursor.execute('''
                           INSERT INTO added_book_ids (book_id)
                           SELECT DISTINCT (cmetadata ->> 'book_id')::integer
                           FROM langchain_pg_embedding
                           LEFT JOIN added_book_ids ON added_book_ids.book_id = (langchain_pg_embedding.cmetadata ->> 'book_id')::integer
                           WHERE added_book_ids.book_id IS NULL;                          
                           ''')
            conn.commit()
                    
    # Select rows to create embeddings from
    query = f'''
            SELECT books.book_id, books.description
            FROM books
            LEFT JOIN added_book_ids ON books.book_id = added_book_ids.book_id
            WHERE added_book_ids.book_id IS NULL
            LIMIT {BATCH_SIZE};
        '''
        
    # Add selected book_ids to added_book_ids
    q = f'''
             INSERT INTO added_book_ids (book_id)
            SELECT books.book_id 
            FROM books
            LEFT JOIN added_book_ids ON books.book_id = added_book_ids.book_id
            WHERE added_book_ids.book_id IS NULL
            LIMIT {BATCH_SIZE};
        '''
        
    with closing(psycopg2.connect(**PARAMS)) as conn:
        with closing(conn.cursor()) as cursor:
            # create table to save embedded books
            cursor.execute(q)
            conn.commit()
        
    df = pd.read_sql(con=CONNECTION_STRING, sql=query)
        
    text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
        
    new_list = []
    for i in range(len(df.index)):
        txt = df['description'][i]
        split_text = text_splitter.split_text(txt)
        for j in range(len(split_text)):
            new_list.append([df['book_id'][i], split_text[j]])
                
    df_new = pd.DataFrame(new_list, columns=df.columns)
        
    loader = DataFrameLoader(df_new, page_content_column='description')
    docs = loader.load()
        
    PGVector.from_documents(
            documents=docs,
            embedding = EMBEDDINGS,
            collection_name = "books_embeddings",
            distance_strategy = DistanceStrategy.COSINE,
            connection_string=CONNECTION_STRING
    )
        
    # return progress
    return get_progress()
        
def get_progress():
    # return progress
    query = '''
        SELECT
        (SELECT COUNT(DISTINCT book_id) FROM added_book_ids) * 100.0 / (SELECT COUNT(DISTINCT book_id) FROM books)
        AS percentage_verified;
    '''
    with closing(psycopg2.connect(**PARAMS)) as conn:
        with closing(conn.cursor()) as cursor:
            # create table to save embedded books
            cursor.execute(query)
            results = cursor.fetchall()
            progress =  float(results[0][0])
    
    return progress
    
    
if __name__ == "__main__":
    load_dotenv('../../.env')
    
    PARAMS = {'user':os.getenv('DB_USERNAME'),
        'password':os.getenv('DB_PASSWORD'),
        'host':os.getenv('DB_HOST'),
        'port':os.getenv('DB_PORT'),
        'database':os.getenv('DB_NAME')}    
    
    # Manually construct the connection string
    CONNECTION_STRING = f"postgresql+psycopg2://{PARAMS['user']}:{PARAMS['password']}@{PARAMS['host']}:{PARAMS['port']}/{PARAMS['database']}"

    EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))
    
    # Create a progress bar
    with tqdm(total=100.0, desc="Progress", position=0, leave=True) as progress_bar:
        progress = get_progress()
        progress_bar.update(progress - progress_bar.n)
        while progress < 100:
            progress = main()
            progress_bar.update(progress - progress_bar.n)
            