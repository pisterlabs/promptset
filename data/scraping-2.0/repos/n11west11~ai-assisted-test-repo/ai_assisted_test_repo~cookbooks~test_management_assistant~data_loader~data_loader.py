from calendar import c
import os
from dotenv import load_dotenv
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from openai.types import file_content
import pymongo
from pymongo import MongoClient
from langchain_community.document_loaders import GitLoader
import fnmatch
import json

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ["MONGO_DATABASE_NAME"]
PATH_TO_PYTHON_FILES = os.environ["PATH_TO_PYTHON_FILES"]

def save_file_to_db(file_path, db):
    with open(file_path, 'r') as file:
        content = file.read()
        db.files.insert_one({"filename": file_path, "content": content})

def load_python_files(path, db, collection_name):

    # Assuming you have a defined RecursiveCharacterTextSplitter and Language
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    # e.g. loading only python files
    loader = GitLoader(
        repo_path=path,
        file_filter=lambda file_path: file_path.endswith(".py"),
        branch="master"
    )
    
    docs = loader.load_and_split(python_splitter)
    for doc in docs:
        doc_json = json.loads(doc.json())
        db[collection_name].insert_one(doc_json)


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

if __name__ == "__main__":
    collection_name = PATH_TO_PYTHON_FILES.split('/')[-1]
    load_python_files(PATH_TO_PYTHON_FILES, db, collection_name)
