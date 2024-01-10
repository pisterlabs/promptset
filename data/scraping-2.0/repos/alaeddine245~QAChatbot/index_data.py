
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

def create_index(documents: List[Document]):
    """
    Create an index from a list of documents.

    Parameters:
    - documents (List[Document]): A list of Document objects to be indexed.

    Returns:
    - Retriever: A retriever object representing the index created from the documents.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) 
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(chunks, embeddings)
    
    # Maximal marginal relevance optimizes for similarity to query and diversity among selected documents
    return db.as_retriever(search_type="mmr")
