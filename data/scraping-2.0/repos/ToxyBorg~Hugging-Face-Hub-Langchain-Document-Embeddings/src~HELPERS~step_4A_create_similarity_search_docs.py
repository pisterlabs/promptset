""" 
    This code defines a function named create_similarity_search_docs that takes in 
    three arguments: 
        - query, 
        - huggingfacehub_api_token, and 
        - path_to_vectorstore. 
    
    The function returns a list of documents that are most similar to the query.

    The function first loads the environment variables from the .env file using the load_dotenv() method. 
    
    It then sets the values of two variables named saving_vectorstore_file_name and saving_vectorstore_directory to the values of the environment variables with the same names. 
    
    It then creates a variable named vectorstore_path which is set to the path of the vectorstore file.

    The function then loads an object of type HuggingFaceHubEmbeddings using the HuggingFaceHubEmbeddings() method. 
    
    It then loads a FAISS vectorstore using the FAISS.load_local() method. 
    
    The function then finds the most similar documents to the query using the faiss.similarity_search()
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain import FAISS

from langchain.schema import Document


load_dotenv()  # Load environment variables from .env file

saving_vectorstore_file_name: str = os.getenv("SAVING_VECTORSTORE_FILE_NAME")
saving_vectorstore_directory: str = os.getenv("SAVING_VECTORSTORE_DIRECTORY")
vectorstore_path = os.path.join(
    saving_vectorstore_directory, saving_vectorstore_file_name + ".faiss"
)


def create_similarity_search_docs(
    query: str,
    huggingfacehub_api_token: str | None = None,
    path_to_vectorstore: str = vectorstore_path,
) -> List[Document]:
    """
    This function takes in three arguments: query, huggingfacehub_api_token, and path_to_vectorstore.
    It returns a list of documents that are most similar to the query.

    Parameters:
        - query (str): The query string.
        - huggingfacehub_api_token (str | None): The Hugging Face Hub API token.
        - path_to_vectorstore (str): The path to the vectorstore file.

    Returns:
        - List[Document]: A list of documents that are most similar to the query.
    """

    # Load the HuggingFaceHubEmbeddings object
    embeddings = HuggingFaceHubEmbeddings(
        huggingfacehub_api_token=huggingfacehub_api_token
    )

    # Load the FAISS vectorstore
    faiss = FAISS.load_local(folder_path=path_to_vectorstore, embeddings=embeddings)

    # Find the most similar documents to the query
    answer_docs = faiss.similarity_search(query, k=4)

    return answer_docs
