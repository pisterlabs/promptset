#! /usr/bin/env python3

"""Load html from files, clean up, split, ingest into Weaviate."""
import os
from dotenv import load_dotenv

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# Load .env
load_dotenv()

# OpenAI Key for OpenAI API
openai_api_key = os.environ['OPENAI_API_KEY']


def ingest_docs():
    """Get documents from web pages."""
    loader = ReadTheDocsLoader("pubhub.devnetcloud.com/media",
                               features="html.parser")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents,
                                        embedding_function,
                                        persist_directory=".chromadb",
                                        collection_name="pyats_docs",)
    vectorstore.persist()


if __name__ == "__main__":
    ingest_docs()
