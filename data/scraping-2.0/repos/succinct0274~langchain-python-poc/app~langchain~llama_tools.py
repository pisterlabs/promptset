import threading
import os
from typing import cast, Dict, Any, List
from langchain_core.pydantic_v1 import Field
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from llama_index.vector_stores import PGVectorStore
from langchain_core.retrievers import BaseRetriever
from llama_index.indices.base import BaseIndex

class LlamaIndexRetriever(BaseRetriever):
    """`LlamaIndex` retriever.

    It is used for the question-answering with sources over
    an LlamaIndex data structure."""

    index: BaseIndex
    """LlamaIndex index to query."""
    query_kwargs: Dict = Field(default_factory=dict)
    """Keyword arguments to pass to the query method."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            from llama_index.indices.base import BaseGPTIndex
            from llama_index.response.schema import Response
        except ImportError:
            raise ImportError(
                "You need to install `pip install llama-index` to use this retriever."
            )
        index = cast(BaseGPTIndex, self.index)

        query_engine = index.as_query_engine(response_mode="no_text", **self.query_kwargs)        
        response = query_engine.query(query)
        response = cast(Response, response)
        # parse source nodes
        docs = []
        for source_node in response.source_nodes:
            metadata = source_node.metadata or {}
            docs.append(
                Document(page_content=source_node.text, metadata=metadata)
            )
        return docs
# Singleton pg vector store
class LlamaIndexPgVectorStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> PGVectorStore:
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = PGVectorStore.from_params(host=os.getenv('POSTGRES_DATABASE_URL'),
                                                              port=os.getenv('POSTGRES_DATABASE_PORT'),
                                                              database=os.getenv('POSTGRES_DATABASE_NAME'),
                                                              user=os.getenv('POSTGRES_DATABASE_USERNAME'),
                                                              password=os.getenv('POSTGRES_DATABASE_PASSWORD'),
                                                              hybrid_search=True)
        
        return cls._instance