import os
import logging
from typing import List, Optional, Dict, Any, Tuple
import faiss

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores.azuresearch import AzureSearch as AzureSearch_TYPE
from langchain.vectorstores.base import VectorStoreRetriever

from app.utils.vectorstore.base import BaseVectorStore
from app.config import Config

logger = logging.getLogger(__name__)


class AzureSearch(BaseVectorStore):

    def __init__(self, config: Config, embeddings: Embeddings):
        """This function initializes the Azure Search Vector Store."""

        super().__init__(config, embeddings)

        self.azure_search_endpoint = config.AZURE_SEARCH_ENDPOINT
        self.azure_search_api_key = config.AZURE_SEARCH_API_KEY


    def _get_langchain_azuresearch(self, index_name: str) -> AzureSearch_TYPE:
        """This function returns a langchain azuresearch object."""

        embedding_fn = self.embeddings.embed_query
        return AzureSearch_TYPE(
            azure_search_endpoint=self.azure_search_endpoint, 
            azure_search_key=self.azure_search_api_key, 
            index_name=index_name,
            embedding_function=embedding_fn
        )
    
    def add_documents(
            self, 
            documents: List[Document], 
            index_name: Optional[str] = None,
            **kwargs: Any
        ) -> None:
        """This function adds documents to the vector store.
        
        Args:
            documents: the documents to add
            index_name: the index name
        Returns:
            none
        """

        # Get langchain azuresearch object
        azure_search = self._get_langchain_azuresearch(index_name)

        # Add documents
        azure_search.add_documents(documents)

    
    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        Args:
            query: the query
            k: the number of results
            filter: the filter
        Returns:
            docs and relevance scores in the range [0, 1].
        """

        # Get langchain azuresearch object
        azure_search = self._get_langchain_azuresearch(index_name)

        # Perform similarity search
        return azure_search.similarity_search_with_relevance_scores(query, k, filter=filter)

