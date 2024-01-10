import json
import os
from typing import Any, Iterable, Optional, List, Tuple
from unittest.mock import patch
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document
from chains.vector_search_chain import VectorSearchChain

class MockVectorStore(VectorStore):
  def similarity_search_with_score(self, query: str, k: int = 5, filter: Optional[dict] = None, namespace: Optional[str] = None) -> List[Tuple[Document, float]]:
    assert query == "How do I open a can of soup?"
    return [
      (Document(page_content="Opening cans of soup.", metadata={}), 0.5),
      (Document(page_content="Opening cans of paint.", metadata={}), 0.4),
    ]
  
  def add_texts(self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any) -> List[str]:
    return super().add_texts(texts, metadatas, **kwargs)
  
  def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
    return super().similarity_search(query, k, **kwargs)

  def from_texts(self, texts: Iterable[str], metadatas: List[dict] | None = None, **kwargs: Any) -> List[str]:
    return super().from_texts(texts, metadatas, **kwargs)  


def test_openai_pinecone_search():
  os.environ.setdefault("OPENAI_API_KEY", "test")
 
  chain = VectorSearchChain(
    query="How do I open a can of {can_type}?",
    embedding_engine="openai",
    embedding_params={"openai_api_key": "test"},
    database="pinecone",
    database_params={"index": "test", "text_key": "text"},
    input_variables=[],
    num_results=10,
  )

  with patch.object(VectorSearchChain, 'vector_store', return_value=MockVectorStore()):
    response = chain._call({"can_type": "soup"})
    results = json.loads(response['results'])
    assert len(results) == 2
    assert results[0]['text'] == "Opening cans of soup."