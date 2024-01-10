import getpass
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

import os
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant


import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_and_process_documents(file_path: Optional[str] = None):
    """
    Loads documents from a given file path, splits them into chunks,
    generates embeddings, and initializes a Qdrant client with these documents.

    Parameters:
    - file_path (str, optional): The path to the file containing the documents.
      Defaults to None. If None, the function will return None.

    Returns:
    - qdrant: An instance of the Qdrant client initialized with the processed documents.
    """
    if file_path is None:
        logging.warning("No file path provided. Returning None.")
        return None

    try:
        loader = TextLoader(f"synthetic_data_3/{file_path}")
        documents = loader.load()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    print("here is the lenght of the docs", len(docs))

    logging.info(f"Documents split into chunks: {docs}")
    
    embeddings = OpenAIEmbeddings()
    
    qdrant_client_url = os.getenv('QDRANT_CLIENT')
    qdrant_api_key = os.getenv('QDRANT_KEY')
    
    if not qdrant_client_url or not qdrant_api_key:
        logging.error("QDRANT_CLIENT or QDRANT_KEY environment variables not set.")
        return None
    
    qdrant = Qdrant.from_documents(
         docs,
         embeddings,
         url=qdrant_client_url,
         prefer_grpc=True,
         api_key=qdrant_api_key,
         collection_name=file_path,
    )
    
    return qdrant



def process_files_in_dict(dict_to_iterate: Dict[str, List[Dict[str, int]]]):
    """
    Iterates over a dictionary of lists of dictionaries, constructs file paths based on the dictionary keys and values,
    reads each file, and then processes the files using the 'load_and_process_documents' function.

    Parameters:
    - dict_to_iterate (Dict[str, List[Dict[str, int]]]): A dictionary where each key corresponds to a list of dictionaries,
      with each sub-dictionary containing a key-value pair used to construct the file path.

    Example of dict_to_iterate:
    {"unstructured": [{"dataset": 1}, {"dataset": 2}, {"dataset": 3}]}
    """
    for key, value in dict_to_iterate.items():
        logging.info(f"Processing key: {key}, Value: {value}")
        for v in value:
            try:
                file_path_key = list(v)[0]  # The key, e.g., 'dataset'
                file_path_value = str(v[file_path_key])  # The value corresponding to the key, e.g., '1'
                file_path = f"{key}_{file_path_key}_{file_path_value}.txt"
                with open("synthetic_data_3/" +file_path, 'r') as file:
                    file_content = file.read()
                    load_and_process_documents(file_path=file_path)
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}")
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")




if __name__ == "__main__":
    dict_to_iterate = {"unstructured": [{"dataset": 1}, {"dataset": 2}, {"dataset": 3}]}
    process_files_in_dict(dict_to_iterate=dict_to_iterate)
