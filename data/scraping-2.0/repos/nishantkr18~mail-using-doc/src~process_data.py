"""
This module contains the code to:
1. Split the data into chunks (sentences).
2. Create vector embeddings of these sentences.
3. Store them in a vectorstore.
"""
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from chromadb.config import Settings
import chromadb


def process_data(docs: List[Document]):
    """
    The function that processes the data.
    """

    # Split into sentences
    source_chunks = []
    splitter = CharacterTextSplitter(
        separator=".", chunk_size=500, chunk_overlap=0)
    for source in docs:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(
                Document(page_content=chunk, metadata=source.metadata))

    print('chunks created: ', len(source_chunks))

    # Create vector embeddings and store in vectorstore.
    print('Creating embeddings...')
    embedding = HuggingFaceEmbeddings()

    print('Creating vectorstore...')

    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./.vectorstore"
    ))
    client.persist()

    # Cleaning up the client
    client.reset()

    vectorstore = Chroma(client=client)
    vectorstore = Chroma.from_documents(
        documents=source_chunks, embedding=embedding, client=client)

    return vectorstore
