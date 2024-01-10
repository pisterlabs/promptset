from spiritualdata_utils import mongo_query_db, mongo_connect_db
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spiritualchat.vectorstores import vector_index
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
from fire import Fire
from ast import literal_eval
import csv
from loguru import logger
from tqdm import tqdm
import time
import pickle
import bson

vector_texts = []

def prepare_embeddings(filepath, dataset: str, chunk_size=1000, chunk_overlap=100, column_to_embed='Description', offset=None, min_length=20, metadata_map={'Experience Type': 'experience_type','Situation Tags': 'situation_tags','Content Tags': 'content_tags','After effects tags': 'after_effects_tags','Date of experience': 'data_experience','Age': 'age','Date reported': 'date_reported','Gender': 'gender','Date Published': 'date_published', 'Authors': 'authors', 'Year': 'date_published', 'Topic': 'topic_tags', 'Topic Tags': 'topic_tags'},
    mongo_field_map={'URL': 'url', 'Name': 'name', 'Language': 'language', 'Authors': 'authors', 'Summary': 'summary'}, delete_previous: bool=False):
    """
    Args:
        - filepath (str): Filepath containing Notion export of documents with 'description' column.
        - dataset (str): Data to be embedded. This is used for namespace. One of 'experiences', 'research', 'hypotheses'
        - chunk_size (int): Size of each chunk (default: 1000).
        - chunk_overlap (int): Overlap between consecutive chunks (default: 20).

    Returns:
        - num_embeddings (int): Number of embeddings created.
    
    Implementation:
        1. Export Notion data to embed each row in Spiritual Research and associate embeddings with metadata (https://www.notion.so/spiritualdata/15b002ffae8e4ccdad4b55a8f619eea8?v=b32c8d520e144c568f1ac60af1cadac4) and Spiritual Experiences (https://www.notion.so/spiritualdata/6c8b84da23054462a3d8bdaa2e1c4968?v=9df4e3c4950744059ba1527d98e85bee).
        2. Embed 'Description' using OpenAI ada 002 embedding with langchain.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    embeddings = OpenAIEmbeddings(chunk_size=chunk_size*2) # chunking here shouldn't be necessary due to text splitting
    num_embeddings = 0
    mongo = mongo_connect_db(database_name='spiritualdata')

    # if delete_previous:
    #     vector_index.delete(deleteAll='true', namespace=dataset)
    #     mongo[dataset].delete_many({})
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        header_column = {header.replace('\ufeff', '').strip(): col_i for col_i, header in enumerate(headers)}
        for i, doc in enumerate(tqdm(reader)):
            try:
                description = doc[header_column['Description']]
                if not description or len(description) < min_length:
                    continue
                mongo_data = document_to_metadata(doc, header_column, mongo_field_map)
                metadata = document_to_metadata(doc, header_column, metadata_map)
                url = norm_id(mongo_data.get('url'))
                if not url:
                    logger.warning('No URL found: '+str(mongo_data))
                    continue
                if not mongo_data.get('name'):
                    logger.warning('No Name found: '+str(doc[header_column['Name']]))
                    continue
                split_metadata = {"source": filepath, "row": i}
                split_docs = text_splitter.split_documents([Document(page_content=description, metadata=split_metadata)])
                if offset and num_embeddings < offset:
                    num_embeddings += len(split_docs)
                    continue
                texts = [doc.page_content for doc in split_docs]
                try:
                    vectors = embeddings.embed_documents(texts)
                except Exception:
                    # Wait 1 minute for rate limiting
                    time.sleep(60)
                if len(texts) != len(vectors):
                    logger.warning(f'Splits texts is of length {len(texts)} but there are {len(vectors)}')
                    continue
                num_embeddings += len(vectors)

                pinecone_docs = []
                mongo_docs = []
                for chunk_index, text in enumerate(texts):
                    # Prepare documents for Pinecone and Mongo databases
                    pinecone_id = url+"_"+str(chunk_index)
                    metadata['chunk_index'] = chunk_index
                    pinecone_docs.append((pinecone_id, vectors[chunk_index], metadata))
                    mongo_doc = dict(mongo_data)
                    mongo_doc['text'] = text
                    mongo_doc['pinecone_id'] = pinecone_id
                    mongo_doc['embedding'] = bson.binary.Binary(pickle.dumps(vectors[chunk_index], protocol=pickle.HIGHEST_PROTOCOL))
                    mongo_docs.append(mongo_doc)
                # Add embeddings to Pinecone with metadata and namespace
                vector_index.upsert(
                    pinecone_docs,
                    namespace=dataset
                )
                result = mongo_query_db(query_type='insert_many', mongo_object=mongo, collection=dataset, to_insert=mongo_docs)
            except Exception:
                logger.exception(f"Reached num_embeddings {num_embeddings}, doc {i}, and URL {url}")
                break
    return num_embeddings

def document_to_metadata(doc, header_column, metadata_map):
    metadata = {}
    for field, metadata_field in metadata_map.items():
        doc_col = header_column.get(field)
        if doc_col is None:
            continue
        value = doc[doc_col]
        if value is not None and value != "":
            try:
                metadata[metadata_field] = literal_eval(value)
            except (SyntaxError, ValueError) as e:
                metadata[metadata_field] = value
    return metadata

def norm_id(id_text):
    if not id_text:
        return None
    return id_text.strip().strip('/').lower()

def embed_csv(df):
    """
    Create OpenAI embeddings for each row in the CSV file
    """
    all_embeddings = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Create the prompt
        prompt = row["Name"] + "\n" + row["Description"] + "\n" + row["URL"] + "\n"
        # Create the embeddings
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[batch_texts],
        )
        # Get the embeddings
        embeddings = response["choices"][0]['data'][0]['embedding']
        all_embeddings.extend(embeddings)
    return embeddings

def remove_duplicates(key='pinecone_id', collections=['experiences']):
    """
    Remove duplicate entries from Mongo by a given field (key).
    """
    mongo = mongo_connect_db(database_name='spiritualdata')
    duplicates_removed = 0

    for collection in collections:
        # Get all pinecone_id
        ids = mongo_query_db(
            query_type="find",
            mongo_object=mongo,
            query={},
            projection={key: 1, '_id': 1},  # Only load pinecone_id
            database='spiritualdata',
            collection=collection
        )
        ids = [doc[key] for doc in ids]

        # Count occurrences
        from collections import Counter
        id_counts = Counter(ids)

        # Find duplicates
        duplicates = [id for id, count in id_counts.items() if count > 1]

        # Remove duplicates
        for duplicate in duplicates:
            # Get all documents with this id
            docs = mongo_query_db(
                query_type="find",
                mongo_object=mongo,
                query={key: duplicate},
                projection={key: 1, '_id': 1},
                database='spiritualdata',
                collection=collection
            )
            # Sort by _id (this is a timestamp in MongoDB)
            docs.sort(key=lambda doc: doc['_id'])
            # Delete all but the first (oldest) document
            for doc in docs[1:]:
                mongo_query_db(
                    query_type="delete_one",
                    mongo_object=mongo,
                    query={'_id': doc['_id']},
                    database='spiritualdata',
                    collection=collection
                )
            duplicates_removed += len(docs) - 1

    print(f'Duplicates removed: {duplicates_removed}')
    return duplicates_removed

if __name__ == "__main__":
    Fire(prepare_embeddings)

"""
Commands to run to prepare embeddings:
python prepare_embeddings.py "Spiritual Experiences 6c8b84da23054462a3d8bdaa2e1c4968.csv" experiences
"""