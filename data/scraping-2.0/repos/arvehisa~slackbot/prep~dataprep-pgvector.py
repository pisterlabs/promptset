from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores.pgvector import PGVector
import os

# split docs
loader = PyPDFDirectoryLoader("./data/")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")

# connect to pgvector & embedding and store data
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=os.environ.get("PGVECTOR_HOST"),
    port="5432",
    database="postgres",
    user="postgres",
    password=os.environ.get("PGVECTOR_PASSWORD"),
)

COLLECTION_NAME = "bedrock_documents"

db = PGVector.from_documents(
    embedding=bedrock_embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING
)