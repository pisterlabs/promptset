import os
import logging
from datetime import datetime
from typing import List

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
import openai
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY

def load_documents(source_dir: str) -> List[Document]:
    '''Extracts pdf(s) into LC Docs'''
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    loader = PyPDFDirectoryLoader(path=source_dir,glob="*.pdf")
    docs = loader.load()
    logging.info(f"load_documents returning {len(docs)}")
    return docs

def main():
    '''Tokenizes, embds and stores embeddings'''
    openai.api_key = os.environ["OPENAI_API_KEY"]

    documents = load_documents(SOURCE_DIRECTORY)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info(f"RecursiveCharacterTextSplitter Start : {datetime.now()}")
    docs = text_splitter.split_documents(documents)
    logging.info("RecursiveCharacterTextSplitter End : {datetime.now()}")
    logging.info(f"RecursiveCharacterTextSplitter Split into {len(docs)} chunks of text")
       
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)
    db.save_local(PERSIST_DIRECTORY)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
