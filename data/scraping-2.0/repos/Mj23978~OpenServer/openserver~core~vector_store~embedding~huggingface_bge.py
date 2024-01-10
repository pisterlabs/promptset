from typing import List
from langchain.embeddings import HuggingFaceBgeEmbeddings

from .base import BaseEmbedding


class HuggingfaceBgeEmbedding(BaseEmbedding):
    
    def __init__(self, device: str = 'cpu', model: str ="BAAI/bge-small-en") -> None:
        self.model: str = model
        self.device: str = device
        model_kwargs: dict[str, str] = {'device': self.device}
        encode_kwargs: dict[str, bool] = {'normalize_embeddings': False}
        self.client = HuggingFaceBgeEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )  # type: ignore

    def get_embeddings(self, texts: List[str]) -> List[List[float]] | dict[str, Exception]:
        try:
            embeds: List[List[float]] = self.client.embed_documents(texts=texts)
            return embeds
        except Exception as exception:
            return {"error": exception}

    def get_embedding(self, text: str) -> List[float] | dict[str, Exception]:
        try:
            embeds: List[float] = self.client.embed_query(text=text)
            return embeds
        except Exception as exception:
            return {"error": exception}
