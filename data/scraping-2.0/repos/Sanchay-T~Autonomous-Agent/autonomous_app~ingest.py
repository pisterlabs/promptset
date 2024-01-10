import json
import logging
import pathlib
from typing import List, Tuple
from langchain.text_splitter import CharacterTextSplitter
import langchain
import wandb
from langchain.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma , Qdrant
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv(".env")


doc_dir = os.path.join("documents" , "iteration_1")

vector_store_path = os.path.join("vector_store")


prompt_file_path = os.path.join("autonomous_app" , "chat_prompt.json")

langchain.llm_cache = SQLiteCache(database_path="llm_cache.db")
logger = logging.getLogger(__name__)



url="https://c48daa38-a7a7-4c89-9cd9-718a1461e4ae.us-east-1-0.aws.cloud.qdrant.io:6333"
api_key_q = "2N5Wlg_y2xwgtJu0Iq5T5uHyesdewC3Vy-VX31YAq-laiQO1tCxAkg"


def ingest_data():

    chunk_size = 600

    chunk_overlap = 200
    
    # Load the documents
    documents = load_documents(doc_dir)
    print("Load Documents " , documents)
    
    # Split the documents into chunks
    chunked_documents = chunk_documents(documents, chunk_size, chunk_overlap)
    
    # Create the vector store with the chunked documents
    vector_store = create_vector_store(chunked_documents)
    return documents, vector_store

def load_documents(data_dir:str) -> List[Document]:
    md_files = list(map(str, pathlib.Path(data_dir).glob("*.md")))
    documents = [
        TextLoader(file_path=file_path , encoding="utf8").load()[0] for file_path in md_files
    ]    
    return documents

def chunk_documents(
    documents: List[Document], chunk_size: int = 600, chunk_overlap=200
) -> List[Document]:
    """Split documents into chunks

    Args:
        documents (List[Document]): A list of documents to split into chunks
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 0.

    Returns:
        List[Document]: A list of chunked documents.
    """
    print("Before chunking " , documents)
    markdown_text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = markdown_text_splitter.split_documents(documents)
    return split_documents

def create_vector_store(chunk_documents) -> Qdrant:

    embedding_function = OpenAIEmbeddings()


    # user_vector_store_path = os.path.join(vector_store_path, user_key)


    # vector_store = Chroma.from_documents(
    #     documents=documents,
    #     embedding=embedding_function,
    #     persist_directory=vector_store_path,
    # )
    #
    # return vector_store

    print("Docs " , chunk_documents)

    vector_store = Qdrant.from_documents(
        chunk_documents,
        embedding_function,
        url=url,
        prefer_grpc=False,
        api_key=api_key_q,
        force_recreate=False,
        collection_name="my_documents",
    )
    return vector_store

# def log_prompt(prompt:dict, run:wandb.run):
#     prompt_artifact = wandb.Artifact(name="chat_prompt", type="prompt")
#     with prompt_artifact.new_file("prompt.json") as f:
#         f.write(json.dumps(prompt))
#     run.log_artifact(prompt_artifact)
#
# def log_dataset(documents:List[Document], run:wandb.run):
#     document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
#     with document_artifact.new_file("document.json") as f:
#         for document in documents:
#             f.write(document.json() + "\n")
#     run.log_artifact(document_artifact)

# def log_index(vector_store_dir:str, run:wandb.run):
#     index_artifact = wandb.Artifact(name="vector_store", type="search_index")
#     index_artifact.add_dir(vector_store_dir)
#     run.log_artifact(index_artifact)
    



# def ingest_and_log_data(
#         docs_dir: str = doc_dir,
#         chunk_size: int = 600,
#         chunk_overlap: int = 200,
#         vector_store_path: str = vector_store_path,
#         prompt_file_path: str = prompt_file_path,
#         wandb_project: str = "AI Agents Hackathon",
#     ):
#     """
#     Ingest documentation data, create a vector store, and log artifacts to W&B.
#     Designed to be used within a Django context.
#     """
#     run = wandb.init(project=wandb_project)  # Move the wandb initialization to this function
#     # user_vector_store_path = os.path.join(vector_store_path, unique_user_key)
#
#
#     # Ingest data
#     documents = ingest_data(
#         docs_dir=docs_dir,
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         vector_store_path=vector_store_path,
#         wandb_project=wandb_project,
#         prompt_file=prompt_file_path,
#     )
#
#     # Log data to wandb
#     log_dataset(documents, run )
#     log_index(vector_store_path, run)
#
#     with open(prompt_file_path, 'r') as f:
#         prompt_data = json.load(f)
#     log_prompt(prompt_data, run)
#
#     # Finish the wandb run
#     run.finish()


ingest_data()

