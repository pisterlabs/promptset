""" 
    This is a Python function that loads similarity search documents from a JSON file. 
    
    The function takes an optional argument similarity_search_docs_file_path which is the path to the JSON file containing the similarity search documents. 
    
    If the argument is not provided, it defaults to the value of similarity_search_docs_file_path which is set to the value of the environment variable SAVING_SIMILARITY_SEARCH_DOCS_DIRECTORY concatenated with the value of the environment variable SAVING_SIMILARITY_SEARCH_DOCS_FILE_NAME with “.json” appended to it.

    The function returns a dictionary containing two keys: “query” and “similarity_search_docs”. 
    
    The value of “query” is a string representing the query used to generate the similarity search documents. 
    The value of “similarity_search_docs” is a list of Document objects which are defined in another module.
"""

import json
import os
from typing import Dict, List, Union

from dotenv import load_dotenv

from langchain.schema import Document

load_dotenv()  # Load environment variables from .env file

save_similarity_search_docs_directory: str = os.getenv(
    "SAVING_SIMILARITY_SEARCH_DOCS_DIRECTORY"
)
save_similarity_search_docs_file_name: str = os.getenv(
    "SAVING_SIMILARITY_SEARCH_DOCS_FILE_NAME"
)

similarity_search_docs_file_path = os.path.join(
    save_similarity_search_docs_directory,
    save_similarity_search_docs_file_name + ".json",
)


def load_similarity_search_docs(
    similarity_search_docs_file_path: str = similarity_search_docs_file_path,
) -> Dict[str, Union[str, List[Document]]]:
    """
    Load similarity search documents from a JSON file.

    Args:
        - similarity_search_docs_file_path (str): The path to the JSON file containing the similarity search documents.
            Defaults to the value of `similarity_search_docs_file_path`.

    Returns:
        - dict: A dictionary containing two keys: "query" and "similarity_search_docs".
            The value of "query" is a string representing the query used to generate
            the similarity search documents. The value of "similarity_search_docs"
            is a list of `Document` objects.
    """

    with open(similarity_search_docs_file_path) as f:
        data = json.load(f)
        query: str = data["query"]
        similarity_search_docs: List[Document] = data["similarity_search_docs"]
    return {"query": query, "similarity_search_docs": similarity_search_docs}
