"""
    This function takes in two arguments: 
    
    load_json_chunks_directory and path_to_ggml_model. 
    
    The first argument is a string that represents the path to the directory containing JSON files with text documents. 
    
    The second argument is a string that represents the path to the LlamaCppEmbeddings model.

    The function loads the LlamaCppEmbeddings model using the provided path, and 
    then iterates through each JSON file in the directory specified by load_json_chunks_directory. 
    
    For each file, it extracts the text content from the JSON and passes it to 
    the LlamaCppEmbeddings model to generate embeddings. 
    
    The embeddings are then added to a list, which is returned by the function.
"""

import os
import sys
import json
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.embeddings import LlamaCppEmbeddings
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.step_2_save_embeddings import save_embeddings


def create_embeddings(
    load_json_chunks_directory: str, path_to_ggml_model: str
) -> List[Embeddings]:
    """
    Creates embeddings for text documents using the LlamaCppEmbeddings model.

    Args:
        - load_json_chunks_directory (str): Path to directory containing JSON files with text documents.
        - path_to_ggml_model (str): Path to the LlamaCppEmbeddings model.

    Returns:
        - List[Embeddings]: A list of embeddings for the text documents.
    """

    # Load LlamaCppEmbeddings object
    embeddings = LlamaCppEmbeddings(model_path=path_to_ggml_model)

    # Embed text from JSON files in directory using LlamaCppEmbeddings
    all_embeddings: list[Embeddings] = []

    for filename in os.listdir(load_json_chunks_directory):
        if filename.endswith(".json"):
            with open(os.path.join(load_json_chunks_directory, filename), "r") as f:
                documents = json.load(f)

            # texts = [doc["chunk_x"] for doc in documents]
            texts: list = []
            for doc in documents:
                for key, value in doc.items():
                    texts.append(value)
                    break

            embeddings_list = embeddings.embed_documents(texts)
            all_embeddings.extend(embeddings_list)

    return all_embeddings


"""################# CALLING THE FUNCTION #################"""

load_dotenv()  # Load environment variables from .env file

print("\n####################### CREATING EMBEDDINGS ########################\n")

load_json_chunks_directory = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS")
path_to_ggml_model: str = os.getenv("PATH_TO_GGML_MODEL")

# Creating the embeddings
embeddings = create_embeddings(
    load_json_chunks_directory=load_json_chunks_directory,
    path_to_ggml_model=path_to_ggml_model,
)

print("\n####################### EMBEDDINGS CREATED ########################\n")

print("\n####################### SAVING EMBEDDINGS ########################\n")
# Saving the embeddings with a specified filename
saving_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME")
saving_embeddings_directory: str = os.getenv("SAVING_EMBEDDINGS_DIRECTORY")

save_embeddings(
    embeddings=embeddings,
    file_name=saving_embeddings_file_name,
    directory_path=saving_embeddings_directory,
)

print("\n####################### EMBEDDINGS SAVED ########################\n")
