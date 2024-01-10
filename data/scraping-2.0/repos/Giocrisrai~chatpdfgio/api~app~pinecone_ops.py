import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import logging


def initialize_pinecone() -> None:
    """
    Initialize Pinecone using the API key and environment specified in the environment variables.
    """
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_env = os.environ.get('PINECONE_API_ENV')
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )


def split_pdf_data(data: str) -> List[str]:
    """
    Split the PDF data into smaller chunks using RecursiveCharacterTextSplitter.

    Parameters:
    data (str): The text data extracted from the PDF.

    Returns:
    list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0
    )
    return text_splitter.split_documents(data)


def generate_and_store_embeddings(data: str, temp_pdf_path: str) -> None:
    """
    Generate and store embeddings for the text data extracted from a PDF.

    Parameters:
    data (str): The text data extracted from the PDF.
    temp_pdf_path (str): The temporary file path where the PDF is stored.
    """
    # Initialize Pinecone if necessary
    initialize_pinecone()

    # Split the PDF data into smaller chunks
    splitted_data = split_pdf_data(data)

    # Set up OpenAI for embedding generation
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Your Pinecone index name
    index_name = os.environ.get('YOUR_INDEX_NAME')

    # Create or update the index in Pinecone
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name, dimension=embeddings.dimension, metric="cosine"
        )

    # Generate and store embeddings
    for i, chunk in enumerate(splitted_data):
        logging.info(f"Processing chunk {i + 1} of {len(splitted_data)}")
        try:
            docsearch = Pinecone.from_texts(
                [chunk.page_content], embeddings, index_name=index_name
            )
        except Exception as e:
            logging.error(f"Error processing chunk {i + 1}: {str(e)}")

    logging.info("All chunks processed successfully")
