# Basics  
import os
from dotenv import load_dotenv
# LLM
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Data Lake
import deeplake
# Langchain
from langchain.schema import Document
from langchain.vectorstores import DeepLake
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    TokenTextSplitter,
)

# Secrets 
load_dotenv()

# Primatives
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()


class PDFprocessing():
    def __init__(self):
        pass   
    
    @staticmethod
    def pdf_pre_processing(repo_path, vs_db_path):
        docs = []
        
        for dirpath, dirnames, filenames in os.walk(repo_path):
            for file in filenames:
                try:
                    print(file)
                    loader = PDFPlumberLoader(repo_path + file)
                    docs.extend(loader.load_and_split(text_splitter=TokenTextSplitter(chunk_size=200, chunk_overlap=0)))

                except Exception as e: 
                    print(e)
                    pass


        try:
            #open(vs_db_path + "dataset_meta.json")
            db = DeepLake(dataset_path=vs_db_path, embedding=embeddings)
            db.add_documents(docs)
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    processor = PDFprocessing()
    processor.pdf_pre_processing("~/.", "./Deeplake/snkl_helper/")   