# from langchain.schema import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader, JSONLoader
# from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.file_processor import FileProcessor
# import faiss
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

class PdfFileProcessor(FileProcessor):
    def document_loader(self,file_path):
        print("pdf file loader")
        loader = PyPDFLoader( file_path=file_path)        # text_content=False)
        return loader.load()

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

