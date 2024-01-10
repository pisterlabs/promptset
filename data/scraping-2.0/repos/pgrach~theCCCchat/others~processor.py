# to be able to loop through the db
import sqlite3
import requests

#to allow concurrence
from concurrent.futures import ThreadPoolExecutor

# to load and split the PDFs into smaller text segments
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

# For creating Semantic Embeddings, storage and retrival
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

#to call API keys from .env
from dotenv import load_dotenv

import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()  # Load variables from .env file
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')

# Connect to SQLite database
conn = sqlite3.connect('links.db')
cursor = conn.cursor()

# Breaking Down PDFs
def process_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check if the request was successful
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve {pdf_url}: {e}")
        return None
        
    loader = PyPDFLoader(pdf_url)
    pages = loader.load_and_split()    
    return pages  # Return pages for further processing

# Pinecone setup
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env
)# Define your Pinecone index name
index_name = "langchain-retrieval-large"

# Create a new index if it doesn't already exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # Assuming the embeddings have a dimension of 1536 (double-checked)
    )

# Connect to the new index
index = pinecone.Index(index_name)

# Embedding Storage
def create_and_store_embeddings(pages):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

        for page_tuple in pages: 
            logging.debug(f"Page tuple: {page_tuple}")  # Log the page_tuple here
            page_content = page_tuple[0]  # Index into page_tuple to get page_content
            metadata = page_tuple[1]  # Index into page_tuple to get metadata
            logging.debug(page_content)
            documents = text_splitter.split_documents(page_content)
            logging.debug(documents)
            embedding = OpenAIEmbeddings(open_ai_key=openai_api_key)
            response = Pinecone.from_documents(documents=documents, embedding=embedding, index_name='langchain-retrieval-large')
            
            # Debugging
            embeddings = embedding.embed_texts([doc.text for doc in documents])
            logging.debug(type(documents))  # Log the type of documents
            logging.debug(documents)  # Log the contents of documents
            if documents:  # Check that documents is not empty
                doc = documents[0]  # Get the first document
                logging.debug(type(doc))  # Log the type of doc
                logging.debug(dir(doc))  # Log the attributes and methods of doc
            logging.debug(embeddings)
            logging.debug(response)
        logging.info("Successful upload")
    except Exception as e:
        logging.error(f"Error: {e}")

def process_and_store(pdf_url):
    pdf_url = pdf_url[0]
    logging.info(f"Processing {pdf_url}")

    pages = process_pdf(pdf_url)
    if pages:
        logging.info(f"Processed {len(pages)} pages from {pdf_url}")
        create_and_store_embeddings(pages)
    else:
        logging.error(f"Failed to process {pdf_url}")

# Iterating PROCESSING through PDFs from 2022 and 2023:
# cursor.execute("SELECT url FROM pdf_links WHERE url LIKE 'https://www.theccc.org.uk/wp-content/uploads/2023%' OR url LIKE 'https://www.theccc.org.uk/wp-content/uploads/2022%'")
cursor.execute("SELECT url FROM pdf_links WHERE id IN (1, 2)")
pdf_urls = cursor.fetchall()

# Temporarily replace ThreadPoolExecutor with sequential processing for debugging
for pdf_url in pdf_urls:
    process_and_store(pdf_url)

conn.close()