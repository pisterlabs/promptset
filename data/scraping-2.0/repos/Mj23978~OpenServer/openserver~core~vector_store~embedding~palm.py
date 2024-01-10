from typing import List
from langchain.embeddings import GooglePalmEmbeddings

from .base import BaseEmbedding


class PalmEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, model="models/text-bison-001") -> None:
        self.model: str = model
        self.api_key: str = api_key
        self.client = GooglePalmEmbeddings(
            model_name=self.model,
            google_api_key=self.api_key,
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
