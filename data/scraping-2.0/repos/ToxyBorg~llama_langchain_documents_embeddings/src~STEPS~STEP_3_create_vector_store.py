"""
    This code defines a function called create_vectorstore_from_json that creates a FAISS index 
    from text embeddings extracted from JSON files in a specified directory. 

    The function takes two arguments: 
        json_files_directory which is the path to the directory containing the JSON files, and 
        model_path which is the path to the model used for generating embeddings. 

    The function: 
        loads the embeddings, 
        reads the JSON files, 
        extracts the text values, 
        creates text embedding pairs, and 
        creates a FAISS index from the pairs.
"""

import json
import os
import sys

from langchain import FAISS
from langchain.embeddings import LlamaCppEmbeddings
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.step_3_loading_embeddings import load_embeddings
from HELPERS.step_3_save_vectorstore import save_vectorstore


def create_vectorstore_from_json(json_files_directory: str, model_path: str) -> FAISS:
    """
    Creates a FAISS index from text embeddings extracted from JSON files in the specified directory.

    Args:
        - json_files_directory (str): Path to directory containing JSON files.
        - model_path (str): Path to model used for generating embeddings.

    Returns:
        - FAISS: FAISS index created from text embedding pairs.
    """

    # Load embeddings
    embeddings = LlamaCppEmbeddings(model_path=model_path)

    load_embeddings_directory: str = os.getenv("SAVING_EMBEDDINGS_DIRECTORY")
    load_embeddings_file_name: str = os.getenv("SAVING_EMBEDDINGS_FILE_NAME")

    embeddings_path = os.path.join(
        load_embeddings_directory, load_embeddings_file_name + ".pkl"
    )

    loaded_embeddings = load_embeddings(file_path=embeddings_path)

    texts: list = []
    for filename in os.listdir(json_files_directory):
        if filename.endswith(".json"):
            with open(os.path.join(json_files_directory, filename), "r") as f:
                chunks = json.load(f)

            for chunk in chunks:
                for key, value in chunk.items():
                    texts.append(value)
                    break

    text_embedding = list(zip(texts, loaded_embeddings))
    faiss = FAISS.from_embeddings(embedding=embeddings, text_embeddings=text_embedding)

    return faiss


"""################# CALLING THE FUNCTION #################"""

load_dotenv()  # Load environment variables from .env file

print("\n####################### CREATING VECTORSTORE ########################\n")

path_to_ggml_model: str = os.getenv("PATH_TO_GGML_MODEL")
json_files_directory = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS")

vectorstore = create_vectorstore_from_json(
    json_files_directory=json_files_directory, model_path=path_to_ggml_model
)

print("\n####################### VECTORSTORE CREATED ########################\n")


print("\n####################### SAVING VECTORSTORE ########################\n")

saving_vectorstore_file_name: str = os.getenv("SAVING_VECTORSTORE_FILE_NAME")
saving_vectorstore_directory: str = os.getenv("SAVING_VECTORSTORE_DIRECTORY")
save_vectorstore(
    vectorstore=vectorstore,
    file_name=saving_vectorstore_file_name,
    directory_path=saving_vectorstore_directory,
)

print("\n####################### VECTORSTORE SAVED ########################\n")
