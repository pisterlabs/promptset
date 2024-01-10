"""
    This code is a Python script that loads documents from a directory, splits them into smaller chunks, and saves the chunks as JSON files. 
    
    The script uses the os, sys, and dotenv modules to handle file paths and environment variables, and the langchain library to load and split the documents. 
    
    The load_documents function takes a directory path as input, iterates through all the files in the directory, determines the file type, loads the document using the appropriate loader, splits the document into smaller chunks using a CharacterTextSplitter, and returns a list of dictionaries containing the document name and chunked data. 
    
    The script then saves the chunked data as JSON files in a specified directory using the save_documents function.
"""

import os
import sys
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter

from typing import List, Dict, Union

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.step_1_save_chunked_docs import save_documents


def load_documents(
    docs_directory_path: str,
) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Load documents from a directory and split them into smaller chunks.

    Args:
        docs_directory_path (str): Path to directory containing documents.

    Returns:
        List[Dict[str, Union[str, List[Dict[str, str]]]]]: A list of dictionaries containing the name and chunked data of each document in the directory. Each dictionary has the following keys:
            - 'name': The name of the document file.
            - 'chunks': A list of dictionaries containing the chunked data of the document. Each dictionary has a key in the format 'chunk_i' (where i is the chunk number) and a value that is the text content of the chunk.
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
            chunk_size=100,
            chunk_overlap=50,
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

load_dotenv()  # Load environment variables from .env file

docs_directory_path = os.getenv("DIRECTORY_DOCUMENTS_TO_LOAD")

# Load documents
loaded_and_chunked_docs = load_documents(docs_directory_path=docs_directory_path)

print("\n####################### DOCUMENTS LOADED ########################\n")


print("\n####################### DOCUMENT CHUNKS LOADED ########################\n")

save_json_chunks_directory = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS")

# Save documents
save_documents(
    documents=loaded_and_chunked_docs,
    save_json_chunks_directory=save_json_chunks_directory,
)


print("\n####################### DOCUMENT CHUNKS SAVED ########################\n")
