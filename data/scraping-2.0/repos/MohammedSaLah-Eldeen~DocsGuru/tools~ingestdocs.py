"""
Ingestion of documentation files.
"""
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tools import logger
from typing import Dict

def ingest_docs(name: str, docsurl: str, storepath: str = None, forced: bool = False) -> None:
    """
    Embeds documentations into a vectorstore using embeddings.

    Args:
        docsurl (str): link to the documentation page.
        storepath (str): Path to the vectorstore.
    """
    # automatic storepath set.
    if storepath is None:
        storepath = f"docsStore/{name}"

    # checking if docs aleady exists.
    try:
        with open('stored_docs.json', 'r') as f:
            stored_docs = json.load(f)
            if name.lower() in stored_docs.keys():
                logger.warning(f"Documentations of {name.lower()} already exists")
                if not forced:
                    return stored_docs.get(name.lower())
                else:
                    logger.info(f"recollecting {name.lower()} with the given url")
            
        with open('stored_docs.json', 'w') as f:
            entry = {
                name.lower(): os.path.abspath(storepath)
            }
            stored_docs.update(entry)
            json.dump(stored_docs, f)
            logger.info(f"created new entry for {name.lower()}")
            
    except FileNotFoundError:
        with open('stored_docs.json', 'w') as f:
            entry = {
                name.lower(): os.path.abspath(storepath)
            }
            json.dump(entry, f)
            logger.info(f"created new entry for {name.lower()}")

    # getting urls.
    response = requests.get(docsurl)
    if response.status_code == 200:
        content = response.text
    else:
        raise ValueError('Incorrect link provided.')

    soup = BeautifulSoup(content, "lxml")
    if ".html" in docsurl:
        docsurl = "/".join(docsurl.split('/')[:-1])

    links = list(dict.fromkeys([urljoin(docsurl, a_tag['href'].split('#')[0]) for a_tag in soup.find_all('a', href=True)]))
    
    # loading
    logger.info("loading documentations...")
    documents = UnstructuredURLLoader(links).load()
    
    # creating chunks
    logger.info("chunking documentations...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    docs = splitter.split_documents(documents=documents)
    
    # embeddings
    embeddings = get_embeddings()

    # vectorstore
    logger.info(f"building vectorstore...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(storepath)

    logger.info(f"vectorstore for {name.lower()} is ready!")
    return os.path.abspath(storepath)


def get_available_docs() -> Dict:
    """
    reads stored_docs.json to retrieve available docs.
    """        
    try:
        with open('stored_docs.json', 'r') as f:
            available_docs = json.load(f)
    except FileNotFoundError:
        with open('stored_docs.json', 'w') as f:
            available_docs = {}
            json.dump(available_docs, f) 
    
    return available_docs


def get_embeddings():
    """
    returns the embeddings object.
    change here to affect all code.
    """ 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
    )
    return embeddings


# loading .env variables
load_dotenv()

