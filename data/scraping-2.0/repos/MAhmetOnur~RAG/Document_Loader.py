import os
import re
import openai
import logging
import pinecone
from dotenv import load_dotenv

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import (CSVLoader,
                                        PyPDFLoader,
                                        UnstructuredURLLoader,
                                        UnstructuredExcelLoader,
                                        UnstructuredPowerPointLoader,
                                        UnstructuredWordDocumentLoader)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


######################################## LOADING DOCUMENT ##############################################################

def load_document(file_path_or_url):
    try:
        # Check if the input is a URL
        if re.match(r'https?://', file_path_or_url):
            loader = UnstructuredURLLoader(urls=[file_path_or_url])
            return loader.load()

        # Determine the file extension
        _, file_extension = os.path.splitext(file_path_or_url)
        file_extension = file_extension.lower()

        # Select the appropriate loader based on the file extension
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path_or_url)

        elif file_extension == '.csv':
            loader = CSVLoader(file_path_or_url)

        elif file_extension in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path_or_url, mode="elements")

        elif file_extension == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path_or_url)

        elif file_extension in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(file_path_or_url)

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return loader.load()

    except Exception as e:
        logging.error(f"Error loading document: {e}")
        return None


def process_and_index_documents(file_path_or_url,
                                pinecone_api_key,
                                pinecone_env,
                                index_name,
                                chunk_size = 1000,
                                chunk_overlap = 0):
    """
    Loads documents, splits them, creates embeddings, and loads them into a Pinecone vector store.

    :param file_path_or_url: Path or URL of the document to process.
    :param pinecone_api_key: API key for Pinecone.
    :param pinecone_env: Environment for Pinecone.
    :param index_name: Name of the Pinecone index to create or use.
    :param chunk_size: Size of each text chunk after splitting. Default is 1000.
    :param chunk_overlap: Overlap size between chunks. Default is 0.
    :return: None
    """

    # Load the document
    data = load_document(file_path_or_url)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                                   chunk_overlap = chunk_overlap)

    texts = text_splitter.split_documents(data)

    pinecone.init(api_key = pinecone_api_key,
                  environment = pinecone_env)

    embeddings = OpenAIEmbeddings()

    Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name = index_name)

    print("Documents processed and indexed successfully.")


process_and_index_documents("",
                            PINECONE_API_KEY,
                            PINECONE_API_ENV,
                            "rag-system")
