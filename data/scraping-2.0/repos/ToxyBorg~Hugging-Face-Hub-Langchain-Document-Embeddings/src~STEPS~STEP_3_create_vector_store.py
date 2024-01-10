"""
    This code creates a vector store from JSON files. The function takes two arguments: 
        - json_files_directory and huggingfacehub_api_token. 
    
    The json_files_directory argument is the directory where the JSON files are stored. 
    The huggingfacehub_api_token argument is the API token for Hugging Face Hub.

    The function loads the HuggingFaceHubEmbeddings object and the embeddings from disk. 
    
    It then creates an empty list to store the texts. 
    
    The function loops through each file in the directory and adds the text to the list of texts. 
    
    It then combines the texts and embeddings into a list of tuples.

    Finally, it creates a FAISS object from the embeddings and text embeddings and returns it.
"""

import json
import os
import sys

from langchain import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from HELPERS.step_3_loading_embeddings import load_embeddings
from HELPERS.step_3_save_vectorstore import save_vectorstore


def create_vectorstore_from_json(
    json_files_directory: str = os.getenv("DIRECTORY_FOR_DOCUMENTS_JSON_CHUNKS"),
    huggingfacehub_api_token: str | None = None,
) -> FAISS:
    """
    This function creates a vector store from JSON files.

    Args:
        json_files_directory (str): The directory where the JSON files are stored.
        huggingfacehub_api_token (str): The API token for Hugging Face Hub.

    Returns:
        FAISS: A FAISS object containing the embeddings.
    """

    # Load the HuggingFaceHubEmbeddings object
    embeddings = HuggingFaceHubEmbeddings(
        huggingfacehub_api_token=huggingfacehub_api_token
    )

    # Load the embeddings from disk
    loaded_embeddings = load_embeddings()

    # Create an empty list to store the texts
    texts: list = []

    # Loop through each file in the directory
    for filename in os.listdir(json_files_directory):
        if filename.endswith(".json"):
            with open(os.path.join(json_files_directory, filename), "r") as f:
                chunks = json.load(f)

            # Loop through each chunk in the file and add the text to the list of texts
            for chunk in chunks:
                for key, value in chunk.items():
                    texts.append(value)
                    break

    # Combine the texts and embeddings into a list of tuples
    text_embedding = list(zip(texts, loaded_embeddings))

    # Create a FAISS object from the embeddings and text embeddings
    faiss = FAISS.from_embeddings(embedding=embeddings, text_embeddings=text_embedding)

    return faiss


"""################# CALLING THE FUNCTION #################"""


print("\n####################### CREATING VECTORSTORE ########################\n")

huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

vectorstore = create_vectorstore_from_json(
    huggingfacehub_api_token=huggingfacehub_api_token
)

print("\n####################### VECTORSTORE CREATED ########################\n")


print("\n####################### SAVING VECTORSTORE ########################\n")

save_vectorstore(vectorstore=vectorstore)

print("\n####################### VECTORSTORE SAVED ########################\n")
