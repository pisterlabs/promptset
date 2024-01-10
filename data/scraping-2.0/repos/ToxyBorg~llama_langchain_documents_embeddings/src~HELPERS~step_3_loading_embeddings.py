"""
    This code defines a function called load_embeddings that loads embeddings from a file using the pickle module. 
        
    The function takes one argument: 
        file_path which is the path to the file containing the embeddings. 
    
    The function: 
        opens the file in binary mode, 
        loads the embeddings using pickle.load(), and 
        returns the embeddings.
"""

import pickle
from langchain.embeddings.base import Embeddings


def load_embeddings(file_path: str) -> Embeddings:
    """
    Loads embeddings from the specified file path using pickle.

    Args:
        - file_path (str): Path to file containing embeddings.

    Returns:
        - Embeddings: Loaded embeddings.
    """

    with open(file_path, "rb") as f:
        embeddings: Embeddings = pickle.load(f)

    return embeddings
