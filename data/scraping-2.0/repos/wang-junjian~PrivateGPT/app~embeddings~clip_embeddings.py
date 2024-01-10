from typing import List
from langchain.embeddings.base import Embeddings


class ClipEmbeddings(Embeddings):
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.model.get_image_features_with_path(text)
