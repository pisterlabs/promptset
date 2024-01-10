from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, JSONLoader


class ChromaDBHandler:
    def __init__(self):
        self.embedding = OpenAIEmbeddings()
        self.persist_directory = '../db'

    def create_db(self):
        documents = []
        loader = TextLoader('../data/return_policy.txt')
        documents.extend(loader.load())
        loader = TextLoader('../data/product_details.txt')
        documents.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            collection_name="data",
            documents=texts,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        return vectordb

    def load_db(self):
        vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        return vectordb

