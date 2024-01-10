# Import LangChain and other libraries
import os
import re

from dotenv import load_dotenv
from langchain.document_loaders import (
    UnstructuredHTMLLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)

# load environment variables
load_dotenv()


def create_document_loader(file):
    # Get the file extension
    ext = os.path.splitext(file)[1]
    # Create a dictionary of file extensions and document loader classes
    loaders = {
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".pdf": UnstructuredPDFLoader,
        ".html": UnstructuredHTMLLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".ppt": UnstructuredPowerPointLoader,
        # Add more as needed
    }
    # Check if the file extension is supported
    if ext in loaders:
        # Return an instance of the corresponding document loader class
        return loaders[ext](file)
    else:
        # Raise an exception if the file extension is not supported
        raise ValueError(f"Unsupported file format: {ext}")


def load_docs(folder):
    # Create an empty list to store the documents
    docs = []
    # Loop through the files in the folder
    for file in os.listdir(folder):
        # Create a document loader object using the factory function
        doc_loader = create_document_loader(os.path.join(folder, file))
        # Load and split the document using the document loader object
        chunks = doc_loader.load_and_split()
        # Append chunks to the list of chunks
        docs.extend(chunks)
    # Return the list of documents
    return docs
