import os
from typing import List
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


def split_data(data: List[Document]) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks.

    This function uses a RecursiveCharacterTextSplitter to split each document's text into smaller chunks. 
    This is useful for processing long documents in systems with constraints on input size.

    Parameters:
    - data (List[Document]): A list of Document objects to be split.

    Returns:
    - List[Document]: A list of Document objects, each containing a portion of the original text.
    """
    # Configure chunk size and overlap
    chunk_size = 1000  # The size of each chunk
    chunk_overlap = 200  # The overlap between chunks

    # Create a TextSplitter object
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split documents into smaller chunks
    return text_splitter.split_documents(data)


def create_embeddings_openai() -> OpenAIEmbeddings:
    """
    Creates an embeddings object using OpenAI's GPT-3 model.

    This function initializes an embeddings object for generating embeddings using OpenAI's API. 
    It requires an API key, which should be set in the environment variables.

    Returns:
    - OpenAIEmbeddings: An instance of OpenAIEmbeddings for generating text embeddings.
    """
    # Set up the OpenAI API key from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY', 'default_api_key')
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Create an OpenAIEmbeddings object
    return OpenAIEmbeddings()


def create_embeddings_open_source(model_name: str) -> SentenceTransformerEmbeddings:
    """
    Creates an embeddings object using the Sentence Transformer model.

    This function initializes an embeddings object using a specified Sentence Transformer model.
    The model name should correspond to one of the pre-trained models available in the Sentence Transformers library.

    Parameters:
    - model_name (str): The name of the Sentence Transformer model to be used.

    Returns:
    - SentenceTransformerEmbeddings: An instance of SentenceTransformerEmbeddings for generating text embeddings.
    """
    # Create a SentenceTransformerEmbeddings object
    return SentenceTransformerEmbeddings(model_name=model_name)
