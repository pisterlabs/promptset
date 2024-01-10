import json
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever


class AstroLoader(BaseLoader):
    def __init__(self, file_path: str):
        """Initialize with a local file path that points to the search.json file."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load json from the local file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        docs = []
        for doc in index:
            metadata = {k: doc[k] for k in ("objectID", "authors", "href", "title")}
            docs.append(Document(page_content=doc["text"], metadata=metadata))
        print(len(docs))
        return docs

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(documents: List[Document], embeddings) -> FAISS:
    return FAISS.from_documents(documents, embeddings)

def ingest_docs() -> ContextualCompressionRetriever:
    """Get documents from web pages."""
    loader = AstroLoader("arxiv_file.json")
    raw_documents = loader.load()

    # Step 1: Split the documents
    split_docs = split_documents(raw_documents)

    # Step 2: Generate embeddings for the documents
    embeddings = OpenAIEmbeddings()

    # Step 3: Create a vector store
    vectorstore = create_vectorstore(split_docs, embeddings)
    # retriever = vectorstore.as_retriever()

    # # Step 4: Create a compressed retriever
    # compressed_retriever = create_compressed_retriever(split_docs, embeddings, retriever)

    return vectorstore


if __name__ == "__main__":
    ingest_docs()
