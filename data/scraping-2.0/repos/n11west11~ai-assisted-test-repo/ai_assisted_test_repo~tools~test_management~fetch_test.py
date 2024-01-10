import os

from dotenv import load_dotenv
from langchain.document_loaders.mongodb import MongodbLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.faiss import FAISS
from pymongo import MongoClient

load_dotenv()


# TODO: I Think this only loads in the tests when the module is loaded, not when the tool is run
# We need to ensure that there is a async task which will periodically update the retriever
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ["MONGO_DATABASE_NAME"]
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]

loader = MongodbLoader(
    connection_string=MONGO_URI,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME
)

docs = loader.load()

db = FAISS.from_documents(docs, OpenAIEmbeddings())

test_retriever_tool = create_retriever_tool(
    db.as_retriever(),
    "TestInfo",
    """Searches and returns relevant information from the Test Database"""
)