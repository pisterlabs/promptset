import openai 
import numpy as np
import dotenv
dotenv.load_dotenv()
import os

def get_text_similarity(first_embedding: list[float], second_embedding: list[float]) -> float:
    return np.dot(first_embedding, second_embedding) # it's recommended to use cosine similarity by openai


def get_embeddings(texts: list[str], model:str ="text-similarity-ada-001") -> dict[str, list[dict[str, list[float]]]]:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return openai.Embedding.create(
    input=texts,
    engine=model)
