import os
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod
import logging
import shutil

from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever

from app.utils.vectorstore import get_vector_store
from app.utils.file.storage import get_storage_client, BLOB_STORAGE_PATERN, is_local_file
from app.config import Config

logger = logging.getLogger(__name__)

DEFAULT_METADATA_SCHEMA = {
    "source": "TEXT",
    "chunk_id": "NUMERIC"
}

class Indexer:
    """This class represents an Indexer."""

    def __init__(self, config: Config):
        """
        Initialize the Indexer.

        Args:
            config: the config object
            vector_store: the vector store
        """

        self.config = config
        self.vector_store = get_vector_store(config)

    def create_index(self, 
                     index_name: str, 
                     metadata_schema: Dict[str, str]=None, 
                     distance_metric: Optional[str]="COSINE"
                     ) -> None:
        """This function creates an index.
            
            The metadata_schema is in the format of:
            {
                "metadata_name": "metadata_type"
            }
            where metadata_type is one of the following:
            - TEXT
            - NUMERIC

            The underlying vector store will automatically create 2 fields automatically:
            1. a field called content which is the text of the document.
            2. a field called content_vector which is the embedding of the document.

            SO, no need to add these 2 fields in the metadata_schema.
        
        Args:
            index_name: the index name
        Returns:
            none
        """
        # TODO: metadata_schema is not used yet.
        self.vector_store.create_index(index_name, DEFAULT_METADATA_SCHEMA, distance_metric)

    def drop_index(self, index_name: str) -> None:
        """This function drops an index.
        
        Args:
            index_name: the index name
        Returns:
            none
        """
        self.vector_store.drop_index(index_name)

    @abstractmethod
    def drop_all_indexes(self) -> None:
        """This function drops all indexes.
        
        Args:
            none
        Returns:
            none
        """

        if self.config.VECTOR_STORE_TYPE == 'faiss':
            # remove the local file
            if os.path.exists(self.config.FAISS_LOCAL_FILE_INDEX):
                logger.info(f"Removing FAISS local file '{self.config.FAISS_LOCAL_FILE_INDEX}'")
                shutil.rmtree(self.config.FAISS_LOCAL_FILE_INDEX)

    @abstractmethod
    def add_document(self, source_url: str, index_name: str, **kwargs: Any) -> None:
        """
        Embed and add the document to the vector store.

        Args:
            source_url: the source url
            index_name: the index name
        Returns:
            none
        """
        
    def check_existing_index(self, index_name: str) -> bool:
        """This function checks if the index exists.
        
        Args:
            index_name: the index name
        Returns:
            none
        """

        return self.vector_store.check_existing_index(index_name)

    def get_retriever(self, index_name: Optional[str] = None) -> VectorStoreRetriever:
        """This function gets the retriever.
        
        Args:
            index_name: the index name
        Returns:
            the retriever
        """

        return self.vector_store.get_retriever(index_name)

class FixedChunkIndexer(Indexer):
    """This class represents a Fixed Chunk Indexer."""

    def __init__(self, config: Config):
        """
        Initialize the Fixed Chunk Indexer.

        Args:
            config: the config object
        """

        super().__init__(config)
        self.chunk_size = config.CHUNKING_STRATEGY_MAX_LENGTH
        self.chunk_overlap = config.CHUNKING_STRATEGY_OVERLAP

        assert self.chunk_size > 0
        assert self.chunk_overlap >= 0

        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def add_document(self, source_url: str, index_name: str, **kwargs: Any) -> None:
        """
        Embed and add the document to the vector store.

        Args:
            source_url: the source url
            index_name: the index name
        Returns:
            none
        """

        try:
            # Check if source url is a file
            if is_local_file(source_url):
                logging.debug(f'Loading document from file {source_url}')
                document = TextLoader(source_url, encoding = 'utf-8').load()
            else:
                logging.debug(f'Loading document from web {source_url}')

                if BLOB_STORAGE_PATERN in source_url:
                    # This is a blob url
                    blob_storage_client = get_storage_client('blob')
                    container, blob_name = blob_storage_client._extract_container_blob_name(source_url)
                    source_url = blob_storage_client.get_blob_sas(container, blob_name)

                document = WebBaseLoader(source_url).load()
        except Exception as e:
            logger.error(e)
            raise e
            
        # TODO: Save the chunks as files as well
        # Split the document into chunks
        chunks = self.text_splitter.split_documents(document)

        # Add metadata to the chunks

        for i, chunk in enumerate(chunks):
            # Add the source url to the metadata
            source_url = source_url.split('?')[0]

            chunk.metadata = {"source": source_url, "chunk_id": i}

        # First load the index from local file if it is a faiss vector store
        if self.config.VECTOR_STORE_TYPE == 'faiss':
            self.vector_store.load_local(self.config.FAISS_LOCAL_FILE_INDEX)

        # Add the chunks to the vector store
        self.vector_store.add_documents(chunks, index_name=index_name, **kwargs)

        # Save the index to local file if it is a faiss vector store
        if self.config.VECTOR_STORE_TYPE == 'faiss':
            self.vector_store.save_local(self.config.FAISS_LOCAL_FILE_INDEX)

        return None
    
    def similarity_search( 
            self, 
            query: str, 
            k: int = 4, 
            filter: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None
        ) -> List[Tuple[Document, float]]:
        """This function performs a similarity search.
        
        """
        # First load the index from local file if it is a faiss vector store
        if self.config.VECTOR_STORE_TYPE == 'faiss':
            self.vector_store.load_local(self.config.FAISS_LOCAL_FILE_INDEX)

        return self.vector_store.similarity_search(query, k, filter=filter, index_name=index_name)
    

    