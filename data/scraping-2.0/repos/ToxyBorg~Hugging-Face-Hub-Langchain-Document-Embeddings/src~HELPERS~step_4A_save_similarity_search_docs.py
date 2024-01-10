""" 
    This function is used to save the similarity search documents in a JSON file. 
    
    The function takes in a list of Document objects, a query string, and two optional parameters for the directory and file name where the output file will be saved.

    The function uses the json.dump() method to write the output to a JSON file. 
    
    If the specified directory does not exist, it is created using os.makedirs().
"""

import json
import os
from typing import List

from dotenv import load_dotenv

from langchain.schema import Document

load_dotenv()  # Load environment variables from .env file


def save_similarity_search_docs(
    similarity_search_docs: List[Document],
    query: str,
    save_similarity_search_docs_directory: str = os.getenv(
        "SAVING_SIMILARITY_SEARCH_DOCS_DIRECTORY"
    ),
    save_similarity_search_docs_file_name: str = os.getenv(
        "SAVING_SIMILARITY_SEARCH_DOCS_FILE_NAME"
    ),
):
    """
    Save similarity search documents in a JSON file.

    Args:
        - similarity_search_docs (List[Document]): A list of Document objects.
        - query (str): The query string.
        - save_similarity_search_docs_directory (str): The directory where the output file will be saved.
        - save_similarity_search_docs_file_name (str): The name of the output file.

    Returns:
        - None
    """

    # Write the output to a JSON file
    if not os.path.exists(save_similarity_search_docs_directory):
        os.makedirs(save_similarity_search_docs_directory)
    output_file = os.path.join(
        save_similarity_search_docs_directory,
        save_similarity_search_docs_file_name + ".json",
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "query": query,
                "similarity_search_docs": [
                    doc.dict() for doc in similarity_search_docs
                ],
            },
            f,
        )
