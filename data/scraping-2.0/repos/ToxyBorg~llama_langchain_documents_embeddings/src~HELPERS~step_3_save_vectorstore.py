"""
    This code defines a function called save_vectorstore that saves a FAISS index as a file at the specified directory path and file name. 

    It imports the os module and the FAISS class from the langchain.vectorstores.faiss module. 

    The function takes three arguments: 
        vectorstore which is the FAISS index to be saved, 
        directory_path which is the path to the directory where the file will be saved, and 
        file_name which is the name of the file to be saved. 
        
    The function:
        creates the directory if it doesn't exist, 
        creates the file path, and 
        saves the FAISS index to the file.
"""

import os
from langchain.vectorstores.faiss import FAISS


def save_vectorstore(vectorstore: FAISS, directory_path: str, file_name: str) -> None:
    """
    Saves a FAISS index as a file at the specified directory path and file name.

    Args:
        - vectorstore (FAISS): FAISS index to be saved.
        - directory_path (str): Path to directory where file will be saved.
        - file_name (str): Name of file to be saved.

    Returns:
        - None
    """

    directory = os.path.join(os.getcwd(), directory_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name + ".faiss")

    vectorstore.save_local(file_path)
