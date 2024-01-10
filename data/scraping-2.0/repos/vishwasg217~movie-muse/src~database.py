from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate
from dotenv import dotenv_values
import os
import json

from utils import get_credentials

def load_data(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def create_db(data: json, db_name: str, active_loop_id: str = "vishwasg217"):

    text = []
    metadata = []

    for key, value in data.items():
        text.append(value['attributes'])
        metadata.append(
            {
                "movie_name": key,
                "movie_id": value['movie_id'],
                "year": value['year'],
                "genres": value['genres'],
                "avg_rating": value['average_rating'],
                "votes": value['num_votes']
            }
        )

    OPEN_AI_API, ACTIVELOOP_TOKEN = get_credentials()

    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)

    dataset_path  = f"hub://{active_loop_id}/{db_name}"
    db = DeepLake.from_texts(text, metadatas=metadata, embedding=embeddings, dataset_path=dataset_path)
    print(f"Accessed dataset at {dataset_path}")
    return db

def load_db(db_name: str, active_loop_id: str = "vishwasg217"):
    dataset_path  = f"hub://{active_loop_id}/{db_name}"
    OPEN_AI_API, ACTIVELOOP_TOKEN = get_credentials()
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only=True)
    return db


if __name__ == "__main__":
    data = load_data("data/processed/final_data.json")
    db = create_db(data, "movie-db")


