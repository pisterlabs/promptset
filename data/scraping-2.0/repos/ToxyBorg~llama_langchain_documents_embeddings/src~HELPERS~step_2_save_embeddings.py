"""
    This function takes in three parameters: 
    
    "embeddings" which is an instance of the "Embeddings" class, 
    "file_name" which is a string representing the name of the file to be saved, and 
    "directory_path" which is a string representing the path to the directory where the file will be saved.

    The function first creates a directory at the specified path if it does not already exist. 
    It then creates a file path by joining the directory path and file name with a ".pkl" extension. 
    Finally, it saves the embeddings object to the binary file using the "pickle" module.
"""


import pickle
import os

from langchain.embeddings.base import Embeddings


def save_embeddings(
    embeddings: Embeddings, file_name: str, directory_path: str
) -> None:
    """
    Save embeddings to a binary file with the specified file name and directory path.

    Args:
        - embeddings (Embeddings): The embeddings to be saved.
        - file_name (str): The name of the file to save the embeddings to.
        - directory_path (str): The path to the directory where the file will be saved.

    Returns:
        - None
    """

    directory = os.path.join(os.getcwd(), directory_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name + ".pkl")

    # Save embeddings to binary file
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
