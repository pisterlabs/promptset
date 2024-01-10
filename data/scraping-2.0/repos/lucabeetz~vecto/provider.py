import os
import openai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def run_openai_embedding(text: str) -> np.ndarray:
    embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(embedding["data"][0]["embedding"])
