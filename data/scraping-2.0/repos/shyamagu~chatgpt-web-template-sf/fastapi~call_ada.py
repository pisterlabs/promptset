
from pydantic import BaseSettings
class Settings(BaseSettings):
    AOAI_EMB_API_KEY :str
    AOAI_EMB_ENDPOINT :str
    AOAI_EMB_MODEL :str

settings = Settings(_env_file=".env")

from logger import logger

from openai import AzureOpenAI
import numpy as np

emb_client = AzureOpenAI(
    api_key = settings.AOAI_EMB_API_KEY,
    api_version = "2023-05-15",
    azure_endpoint =settings.AOAI_EMB_ENDPOINT
)

def generate_embeddings(text, model=settings.AOAI_EMB_MODEL): # model = "deployment_name"
    return emb_client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def encode_content(content):
    return generate_embeddings(content)

def get_similarity(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity

def encode_contents (contents):
    # contentをループ処理
    for i in range(len(contents)):
        embedding = generate_embeddings(contents[i]["content"])
        contents[i]["embedding"] = embedding
        logger.debug(str(i+1)+"/"+str(len(contents))+": embedded")
    return contents

def search_contents (message,embedding_data):

    embedding = generate_embeddings(message)
    for i in range(len(embedding_data)):
        embedding_data[i]["similarity"] = cosine_similarity(embedding, embedding_data[i]["embedding"])
    
    # similarityでソート
    embedding_data = sorted(embedding_data, key=lambda x: x["similarity"], reverse=True)

    # embedding_dataの上位5件のidを配列として取得
    top5ids = [embedding_data[i]["id"] for i in range(5)]

    return top5ids

def search_contents_simple (message,embedding_data):
    embedding = generate_embeddings(message)
    for i in range(len(embedding_data)):
        embedding_data[i]["similarity"] = cosine_similarity(embedding, embedding_data[i]["embedding"])
    
    # similarityでソート
    embedding_data = sorted(embedding_data, key=lambda x: x["similarity"], reverse=True)

    return embedding_data
