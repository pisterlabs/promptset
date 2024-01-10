from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


def create_index(docs: List[Document]) -> VectorStoreRetriever:
    """ Creates a vectorstore index from a list of Document objects.

    Args:
        docs: List of Document objects.

    Returns:
        A vectorstore index. It searches the most similar document to the given query but with
        the help of MMR it also tries to find the most diverse document to the given query.
        
    """
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
        
    ).from_documents(docs)

    return index.vectorstore.as_retriever(search_type='mmr')
