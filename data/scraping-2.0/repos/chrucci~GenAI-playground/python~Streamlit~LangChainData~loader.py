from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st


class PdfLoader:
    persist_directory = "./docs/chroma/"

    def __init__(self) -> None:
        self.embedding = OpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory=PdfLoader.persist_directory,
            embedding_function=self.embedding,
        )

    def load_pdf(self, file_path):
        print(f" this is the path: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return docs

    def split_doc(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        splits = text_splitter.split_documents(docs)
        return splits

    def add_to_db(self, splits):
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding,
            persist_directory=PdfLoader.persist_directory,
        )
        vectordb.persist()
        return vectordb

    def get_db(self):
        return self.vectordb

    @st.cache_resource
    def load(_self, file_path):
        doc = _self.load_pdf(file_path)
        splits = _self.split_doc(doc)
        return _self.add_to_db(splits)


# Load
# pdf_file_name="test.pdf"

# pdf_length = len(docs)
# pdf_text = f"number of pages in pdf = {pdf_length}"
# print(pdf_text)

# #Split
# split_len = len(splits)
# split_text = f"number of splits in pdf = {split_len}"
# print(split_text)

# #Storage


# vectordb = load()

# vector_db_len = vectordb._collection.count()
# embedding_text = f"number of embeddings from pdf = {vector_db_len}"
# print(embedding_text)
