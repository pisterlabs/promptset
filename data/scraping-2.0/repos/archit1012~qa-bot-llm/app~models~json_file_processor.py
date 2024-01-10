# from langchain.schema import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.file_processor import FileProcessor
# import faiss
from langchain.vectorstores import FAISS

class JsonFileProcessor(FileProcessor):
    def document_loader(self, file_path):
        try:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False)
            return loader.load()
        except Exception as e:
            print("Something went wrong while performing 'document_loader' operations", e)

    def text_splitter(self,documents):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            return text_splitter.split_documents(documents)
        except Exception as e:
            print("Something went wrong while performing 'text_splitter' operations", e)


    def prepare_vectordb(self,docs,embeddings):
        try:
            vector_db = FAISS.from_documents(docs, embeddings)
            return vector_db
        except Exception as e:
            print("Something went wrong while performing 'prepare_vectordb' operations", e)

