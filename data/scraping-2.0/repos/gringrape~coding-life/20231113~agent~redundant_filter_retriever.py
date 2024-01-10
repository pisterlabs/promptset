from typing import Any, Coroutine, Dict, List, Optional
from langchain.callbacks.manager import Callbacks
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str) -> List[Document]:
        embedding = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            lambda_mult=0.8,
        )

    def aget_relevant_documents(self) -> Coroutine[Any, Any, List[Document]]:
        return []
