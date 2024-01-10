import os
import sys
import glob
import sqlite3
import requests
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma

from datetime import datetime
import constants

def scrape_webpages(urls):
    scraped_data = []
    print("Scraping data and storing urls: ", urls)
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        scraped_data.append(text)
    return scraped_data

def store_scraped_data(scraped_data):
    for i, text in enumerate(scraped_data):
        with open(f'web_content/page{i}.txt', 'w') as f:
            f.write(text)
    print("Stored data in web_content folder")

def initialize_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations
        (id INTEGER PRIMARY KEY, date TEXT, tags TEXT, title TEXT, vector_id TEXT)
    ''')
    conn.commit()
    print("sqlite Database initialized\n")
    return conn 

def insert_conversation(date, tags, title, vector_id):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (date, tags, title, vector_id)
        VALUES (?, ?, ?, ?)
    ''', (date, tags, title, vector_id))
    conn.commit()
    conn.close()

class SQLTableLoader():
    def __init__(self, conn, table_name):
        self.conn = conn
        self.table_name = table_name
    def load(self):
        with self.conn as conn:
            c = conn.cursor()
            result = c.execute(f"SELECT * FROM {self.table_name}")
            rows = result.fetchall()
            if len(rows) == 0:
                return []
            for row in rows:
                yield type('Document', (object,), {'page_content': row[4], 'metadata': {'tags': row[2]}})

def get_index(PERSIST):
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        from langchain.indexes.vectorstore import VectorStoreIndexWrapper
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        print("Loading all files into loaders...\n")
        return create_index_from_loaders(PERSIST)

def create_index_from_loaders(PERSIST):
    dir_path = 'project/'
    file_types = ['html', 'js', 'json', 'txt', 'md', 'toml']
    loaders = []
    for file_type in file_types:
        glob_pattern = '**/*.' + file_type
        for file_path in glob.glob(os.path.join(dir_path, glob_pattern), recursive=True):
            if "node_modules" not in file_path:
                if file_type == 'html':
                    print("Loading", file_path, "as unstructured HTML")
                    loader = UnstructuredHTMLLoader(file_path)
                else:
                    print("Loading", file_path, "as text")
                    loader = TextLoader(file_path)
                loaders.append(loader)

    web_content_loader = DirectoryLoader('web_content/', glob='**/*.txt',
                                        show_progress=True, use_multithreading=True,
                                        loader_cls=TextLoader)
    loaders.append(web_content_loader)

    if '--no-history' not in sys.argv:
        conn = initialize_db()
        history_loader = SQLTableLoader(conn, 'conversations')
        loaders.append(history_loader)
    else:
        print("Not loading history\n")

    total_docs = sum(1 for loader in loaders for _ in loader.load())
    print("Creating index of", total_docs, "documents..\n")
    if PERSIST:
        return VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders(loaders)
    else:
        return VectorstoreIndexCreator().from_loaders(loaders)

def create_chain(index):
    modelName =  sys.argv[sys.argv.index('-m') + 1] if '-m' in sys.argv else "gpt-3.5-turbo"
    numberOfResults = int(sys.argv[sys.argv.index('-k') + 1]) if '-k' in sys.argv else 5
    print("Creating chain with model:", modelName, "\n")
    print("Number of results:", numberOfResults, "\n")
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=modelName),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": numberOfResults}),
    )

if __name__ == "__main__":
    if '--no-history' not in sys.argv:
        initialize_db()
    os.environ["OPENAI_API_KEY"] = constants.APIKEY
    query = sys.argv[sys.argv.index('-q') + 1] if '-q' in sys.argv else None
    PERSIST = sys.argv.index('--persist') if '--persist' in sys.argv else False
    print("Persisting index:", PERSIST, "\n")

    urls = []
    with open('urls.txt', 'r') as f:
        for line in f:
            urls.append(line.strip())
    scraped_data = scrape_webpages(urls)
    store_scraped_data(scraped_data)
    index = get_index(PERSIST)
    print("Index created\n")
    chain = create_chain(index)

    print("Ready to chat!\n")
    while True:
        if query:
            user_input = query
            query = None
        else:
            user_input = input(">>>")
        if user_input.lower() == 'exit':
            break

        if '--no-history' not in sys.argv:
            timestampNow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_conversation(date=timestampNow, tags='example', title='Example Conversation', vector_id=user_input)
        result = chain.run(user_input)
        if '--no-history' not in sys.argv:
            insert_conversation(date=timestampNow, tags='example', title='Example Conversation', vector_id=result)
        print(result)