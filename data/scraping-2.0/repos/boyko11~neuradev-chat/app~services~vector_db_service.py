from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VST


class VectorDBService:
    _vector_store_instance: VST = None

    @classmethod
    def add_docs(cls, documents=None, embeddings=OpenAIEmbeddings()):
        if cls._vector_store_instance is None:
            cls._vector_store_instance = FAISS.from_documents(documents, embeddings)
        else:
            cls._vector_store_instance.add_documents(documents)

    @classmethod
    def db_as_retriever(cls):
        if cls._vector_store_instance is None:
            raise Exception("No documents have been added to the database yet")
        return cls._vector_store_instance.as_retriever()
