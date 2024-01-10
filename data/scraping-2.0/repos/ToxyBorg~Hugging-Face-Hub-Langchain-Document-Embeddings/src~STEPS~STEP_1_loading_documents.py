""" 
    This code is a Python function that loads documents from a directory and returns a list of dictionaries containing the name of each document and its chunks. 
    
    The function uses the langchain package to load documents from different file types such as pdf or unstructured files. 
    
    It then splits each document into smaller chunks using the CharacterTextSplitter class from the same package. 
    
    The chunks are then saved in a dictionary format with keys such as “chunk_1”, “chunk_2”, etc. 
    
    Finally, the function returns a list of dictionaries containing the name of each document and its chunks.
"""

import os
import sys
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter

from typing import List, Dict, Union

load_dotenv()  # Load environment variables from .env file

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.step_1_save_chunked_docs import save_documents


def load_documents(
    docs_directory_path: str = os.getenv("DIRECTORY_DOCUMENTS_TO_LOAD"),
) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Load documents from a directory and return a list of dictionaries containing the name of each document and its chunks.

    Args:
        docs_directory_path (str): The path to the directory containing the documents to load.

    Returns:
        List[Dict[str, Union[str, List[Dict[str, str]]]]]: A list of dictionaries containing the name of each document and its chunks.
    """

    result = []

    # Iterate through all the files in the directory
    for file_name in os.listdir(docs_directory_path):
        file_path = os.path.join(docs_directory_path, file_name)

        # Determine loader based on file type
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path=file_path)
        else:
            loader = UnstructuredFileLoader(file_path=file_path)

        # Load document
        document = loader.load()

        # Split document into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=200,
            chunk_overlap=75,
            length_function=len,
        )

        chunks = [
            {"chunk_" + str(i + 1): chunk.page_content}
            for i, chunk in enumerate(text_splitter.split_documents(documents=document))
        ]

        # Add document name and chunked data to result list
        file_name = os.path.splitext(file_name)[0]
        result.append({"name": file_name, "chunks": chunks})

    return result


"""################# CALLING THE FUNCTION #################"""

print("\n####################### LOADING DOCUMENTS ########################\n")

# Load documents
loaded_and_chunked_docs = load_documents()

print("\n####################### DOCUMENTS LOADED ########################\n")


print("\n####################### DOCUMENT CHUNKS LOADED ########################\n")

# Save documents
save_documents(documents=loaded_and_chunked_docs)

print("\n####################### DOCUMENT CHUNKS SAVED ########################\n")
