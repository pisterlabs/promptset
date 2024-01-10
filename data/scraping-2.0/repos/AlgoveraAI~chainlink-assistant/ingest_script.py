import os
import faiss
import pickle
import requests
import argparse
from dotenv import load_dotenv
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingest.docs import scrap_docs
from ingest.blogs import scrap_blogs
from ingest.education import scrap_education_docs
from ingest.stackoverflow import scrap_stackoverflow
from ingest.data import scrap_data
from ingest.chain_link import scrap_chain_link
from config import get_logger, DATA_DIR
from chat.utils import CustomeSplitter
from fastapi import HTTPException

logger = get_logger(__name__)

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = "http://localhost"
LOGIN_URL = "https://stackoverflow.com/users/login"
OAUTH_URL = f"https://stackoverflow.com/oauth?client_id={CLIENT_ID}&scope=read_inbox&redirect_uri={REDIRECT_URI}"
TOKEN_URL = "https://stackoverflow.com/oauth/access_token"
date_str = datetime.now().strftime(
    "%Y-%m-%d"
)  # Get today's date as "YYYY-MM-DD" format


def get_access_token():
    # Create a session to maintain cookies
    session = requests.Session()

    # Try to access OAuth URL
    response = session.get(OAUTH_URL, allow_redirects=False)
    logger.info(f"Initial get code response: {response.status_code}")
    logger.info(
        f"Initial get code response headers: {response.headers.get('location')}"
    )
    if (
        response.status_code == 302
        and "location" in response.headers
        and "/users/login" in response.headers["location"]
    ):
        # We're being redirected to login
        login_data = {
            "email": os.getenv("SO_LOGIN_EMAIL"),
            "password": os.getenv("SO_LOGIN_PASSWORD"),
        }
        response = session.post(LOGIN_URL, data=login_data)
        logger.info(f"Login response: {response.status_code}")

        # After login, try accessing OAuth URL again
        response = session.get(OAUTH_URL, allow_redirects=False)
        logger.info(f"Second get code response: {response.status_code}")
        logger.info(
            f"Second get code response headers: {response.headers.get('location')}"
        )

    # Extract the intermediary code
    if response.status_code == 302 and "code=" in response.headers["location"]:
        code = response.headers["location"].split("code=")[1].split("&")[0]

        # Exchange the intermediary code for the access token
        token_data = {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "redirect_uri": REDIRECT_URI,
        }
        token_response = requests.post(TOKEN_URL, data=token_data)
        if token_response.status_code == 200:
            response_content = token_response.content.decode(
                "utf-8"
            )  # Convert byte-string to string
            actual_token = response_content.split("access_token=")[1].split("&")[0]

            return {"access_token": actual_token}

    raise HTTPException(
        status_code=400, detail="Unable to fetch the intermediary code."
    )


def ingest_task():
    # Get access token
    access_token = get_access_token()

    # Parse technical documentation
    docs_documents = scrap_docs()

    # Parse blog posts
    blog_urls = scrap_blogs()

    # Parse chainlink.education
    chainlink_education_documents = scrap_education_docs()

    # Parse stackoverflow
    stackoverflow_documents = scrap_stackoverflow(access_token["access_token"])

    # Parse data.chain.link
    data_documents = scrap_data()

    # Parse chain.link
    chain_link_documents, chain_link_youtube_documents = scrap_chain_link()

    # Combine all documents into one list (except data)
    documents = []
    documents_count = 0
    documents_count += len(docs_documents)
    documents.extend(docs_documents)
    documents_count += len(blog_urls)
    documents.extend(blog_urls)
    documents_count += len(chainlink_education_documents)
    documents.extend(chainlink_education_documents)
    documents_count += len(stackoverflow_documents)
    documents.extend(stackoverflow_documents)
    documents_count += len(chain_link_documents)
    documents.extend(chain_link_documents)
    documents_count += len(chain_link_youtube_documents)
    documents.extend(chain_link_youtube_documents)

    # Log the number of documents
    logger.info(f"Docs: {len(docs_documents)}")
    logger.info(f"Blogs: {len(blog_urls)}")
    logger.info(f"Education: {len(chainlink_education_documents)}")
    logger.info(f"Stackoverflow: {len(stackoverflow_documents)}")
    logger.info(f"Data: {len(data_documents)}")
    logger.info(f"Chain Link: {len(chain_link_documents)}")
    logger.info(f"Chain Link Youtube: {len(chain_link_youtube_documents)}")

    # Log the total number of documents
    logger.info(f"Total: {len(documents)}")

    # For saving documents:
    with open(f"{DATA_DIR}/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    # Split documents into chunks for 16k model
    full_doc_splitter = CustomeSplitter()
    chunked_full_documents = full_doc_splitter.split(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    # Create vectorstore for all documents
    vectorstore_all = FAISS.from_documents(split_docs, embedding=OpenAIEmbeddings())

    # Split documents into chunks using datadocs
    split_docs_data = splitter.split_documents(data_documents)

    # Create vectorstore for datadocs
    vectorstore_data = FAISS.from_documents(
        split_docs_data, embedding=OpenAIEmbeddings()
    )

    # Save vectorstores to disk
    faiss.write_index(vectorstore_all.index, f"{DATA_DIR}/docs_all.index")
    vectorstore_all.index = None
    with open(f"{DATA_DIR}/faiss_store_all.pkl", "wb") as f:
        pickle.dump(vectorstore_all, f)

    # Save vectorstore_data
    faiss.write_index(vectorstore_data.index, f"{DATA_DIR}/docs_data.index")
    vectorstore_data.index = None
    with open(f"{DATA_DIR}/faiss_store_data.pkl", "wb") as f:
        pickle.dump(vectorstore_data, f)

    logger.info("Done")


if __name__ == "__main__":
    ingest_task()
