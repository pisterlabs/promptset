from typing import Any, Dict, List, Optional
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document

class RedudantFilterRetriever(BaseRetriever):
  embeddings: Embeddings
  chroma: Chroma

  def get_relevant_documents(self, query: str) -> List[Document]:
    """
      Calculate embeddings for the query and return the most similar documents
    """
    emb = self.embeddings.embed_query(query)
    return self.chroma.max_marginal_relevance_search_by_vector(
      emb,
      lambda_mult=0.8 # tollerance for redundancy
    )
  async def aget_relevant_documents(self):
    return []