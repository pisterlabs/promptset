import os
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader, UnstructuredURLLoader
from langchain.schema import Document
from typing import List


def load_file(file_path: str) -> List[Document]:
    """
    Load documents from a specified file based on its extension.

    Parameters:
    - file_path (str): The path to the file to be loaded.

    Returns:
    - List[Document]: A list of Document objects representing the content of the loaded file.
    """
    extension = os.path.splitext(file_path)[-1]
    if extension == ".pdf":
        print("triggered.")
        return PyPDFLoader(file_path=file_path).load()
    elif extension == ".txt":
        return TextLoader(file_path=file_path).load()
    elif extension == ".html":
        return UnstructuredHTMLLoader(file_path=file_path).load()
    
def load_website(url: str) -> List[Document]:
    """
    Load documents from a specified website URL.

    Parameters:
    - url (str): The URL of the website to be loaded.

    Returns:
    - List[Document]: A list of Document objects representing the content of the loaded website.
    """
    return UnstructuredURLLoader(urls=[url]).load()
