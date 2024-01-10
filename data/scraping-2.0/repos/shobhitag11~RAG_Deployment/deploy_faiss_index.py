import os
from typing import List, Union
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel

# Create a FastAPI application
app = FastAPI()

# Set the path for the local downloaded embed model path:
# We may have models like:
# 1. all-mpnet-base-v2
# 2. all-distilroberta-v1
EMBED_PATH = "<EMBED_MODEL_LOCAL_PATH>"

class LocalHuggingFaceEmbeddings(Embeddings):

    def __init__(self, model_id):
        self.model = SentenceTransformer(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        :param texts: The list of texts to embed
        :return: List of Embeddings, one for each text
        """
        embeddings = self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        :param texts: The text to embed
        :return: Embeddings for the text
        """
        embeddings = self.model.encode(text)
        return List(map(float, embeddings))

embeddings = LocalHuggingFaceEmbeddings(EMBED_PATH)

async def call_model(index_path: str, query: str):
    db = FAISS.load_local(index_path, embeddings)
    retrieved_context = db.max_marginal_relevance_search(query)
    return {
        "context": retrieved_context
    }
