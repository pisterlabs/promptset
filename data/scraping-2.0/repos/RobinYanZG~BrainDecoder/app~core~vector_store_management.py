import os
from langchain.vectorstores import FAISS
from app.helper.path import path
from app.core.model.embedding_model import EmbeddingModel


class VectorStoreManagement: 
    vector_store: FAISS
    
    def load(self, index_key: str, embedding: EmbeddingModel) -> object:
        key = os.path.join(path.VECTOR_STORE_PATH, index_key)
        print(key)
        self.vector_store = FAISS.load_local(key, embedding)
        
    def similarity_search_with_score(self, query: str, k: int) -> list:
        return self.vector_store.similarity_search_with_score(query, k)
    
    def similarity_search_under_score(self, query: str, score: int, k:int=10) -> list:
        result = []
        records = self.similarity_search_with_score(query, k)
        for record in records:
            if record[1] < score:
                result.append(record[0])
        
        return result