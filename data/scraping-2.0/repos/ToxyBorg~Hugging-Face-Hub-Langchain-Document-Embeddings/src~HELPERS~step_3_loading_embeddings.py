"""
    This code defines a function called load_embeddings that loads embeddings from a file using the pickle module. 
        
    The function takes one argument: 
        file_path which is the path to the file containing the embeddings. 
    
    The function: 
        opens the file in binary mode, 
        loads the embeddings using pickle.load(), and 
        returns the embeddings.
"""
import os

import pickle
from langchain.embeddings.base import Embeddings

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

load_embeddings_directory: str = os.getenv("SAVING_EMBEDDINGS_DIRECTORY")
load_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME")

embeddings_path = os.path.join(
    load_embeddings_directory, load_embeddings_file_name + ".pkl"
)


def load_embeddings(
    embeddings_path: str = embeddings_path,
) -> Embeddings:
    """
    Loads embeddings from the specified file path using pickle.

    Args:
        - embeddings_path (str): Path to file containing embeddings.

    Returns:
        - Embeddings: Loaded embeddings.
    """

    with open(embeddings_path, "rb") as f:
        embeddings: Embeddings = pickle.load(f)

    return embeddings
