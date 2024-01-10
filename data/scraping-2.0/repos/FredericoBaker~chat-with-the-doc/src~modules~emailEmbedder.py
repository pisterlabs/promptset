import os
import pickle
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmailEmbedder:

    def __init__(self):
        self.PATH = "embeddings"
        self.createEmbeddingsDir()

    def createEmbeddingsDir(self):
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def storeDocEmbeddings(self, originalFileName):

        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len
        )

        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["sender"] = record.get("sender")
            metadata["date"] = record.get("date")
            metadata["subject"] = record.get("subject")
            return metadata

        loader = JSONLoader(file_path=originalFileName, 
                            jq_schema='.messages[]', 
                            content_key='body',
                            metadata_func=metadata_func)
        data = loader.load()

        embeddings = OpenAIEmbeddings()

        vectors = FAISS.from_documents(data, embeddings)
        os.remove(originalFileName)

        with open(f"{self.PATH}/{originalFileName}.pkl", "wb") as file:
            pickle.dump(vectors, file)

    def getDocEmbeddings(self, originalFileName):
        if not os.path.isfile(f"{self.PATH}/{originalFileName}.pkl"):
            self.storeDocEmbeddings(originalFileName)
        
        with open(f"{self.PATH}/{originalFileName}.pkl", "rb") as file:
            vectors = pickle.load(file)

        return vectors