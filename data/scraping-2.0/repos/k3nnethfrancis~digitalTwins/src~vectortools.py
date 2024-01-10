from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader, WebBaseLoader
from pathlib import Path

from dotenv import load_dotenv
import os

# Get the current project directory
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

# Load .botenv file from the project's root directory
load_dotenv(os.path.join(THIS_DIR, '../bot.env'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Define the path to the document
DOC_PATH = str(Path(".").resolve() / "docs" / "infos.txt")

def load_api_key():
    # Get the current project directory
    THIS_DIR = os.path.dirname(os.path.realpath(__file__))
    # Load .botenv file from the project's root directory
    load_dotenv(os.path.join(THIS_DIR, '../bot.env'))
    return os.getenv("OPENAI_API_KEY")

def create_text_splitter():
    return CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

def process_documents(loader_class, doc_path, collection_name, persist_directory):
    loader = loader_class(doc_path)
    documents = loader.load()
    text_splitter = create_text_splitter()
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        texts,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    retriever = db.as_retriever()
    llm = OpenAI(api_key=load_api_key(), temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever
    )
    db.persist()
    return db, retriever, chain

# This will only run if the script is run directly (not when imported)
if __name__ == '__main__':
    info_db, info_retriever, info_chain = process_documents(TextLoader, DOC_PATH, 'infos', './db/infos')
    #db2, retriever2 = process_documents(WebBaseLoader, "https://www.k3nnethfrancis.com/gpt-out-of-the-box/", 'latest', './db/latest')