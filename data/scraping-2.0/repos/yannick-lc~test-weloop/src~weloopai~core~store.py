"""
Contains classes and functions to store and retrieve information
"""

import logging

from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from weloopai.config.configuration import KNOWLEDGE_BASE_FOLDER, VECTORSTORE_FOLDER

logger = logging.getLogger("store")


class KnowledgeStore:
    """
    Class responsible for storing information in the form of vectors,
    and retrieving relevant documents based on a query.
    """
    N_DOCUMENTS_TO_RETRIEVE = 2

    def __init__(self):
        """Initialize the store and select the embedding function"""
        self.documents = []
        self.embedding = OpenAIEmbeddings()

    def load(self) -> list[Document]:
        """Read the documents from the knowledge base folder"""
        loader = DirectoryLoader(str(KNOWLEDGE_BASE_FOLDER), glob="**/*.txt")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents")
        self.documents = docs
        return self.documents

    def store(self) -> None:
        """Compute the embeddings of the documents and persist them as vectors"""
        if self.documents == []:
            raise ValueError("No document loaded")
        Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding,
            persist_directory=str(VECTORSTORE_FOLDER)
        )
        logger.info(f"Embedded documents and stored as vectors in {VECTORSTORE_FOLDER}")
    
    def get_vectorstore(self) -> VectorStoreRetriever:
        """Load the vectorstore and return a retriever"""
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_FOLDER),
            embedding_function=self.embedding
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.N_DOCUMENTS_TO_RETRIEVE})
        return retriever

def store_as_vectors() -> None:
    """
    Instantiate a store, load the documents from the knowledge base folder,
    compute the embeddings and store them as vectors
    """
    store = KnowledgeStore()
    store.load()
    store.store()
