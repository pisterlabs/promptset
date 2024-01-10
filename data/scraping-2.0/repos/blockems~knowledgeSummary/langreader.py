import os
import json

#Env Mgt
from dotenv import load_dotenv
from pypdf import PdfReader
from sympy import per
#from torch import embedding


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
data_conn = os.getenv('DATA_CONNECTION_STRING', './data/database.db')
data_processed= os.getenv('DATA_PROCESSED_DIR', './source/processed')
data_pre_processed= os.getenv('DATA_PRE_PROCESSED_DIR', './source/preprocessed')
data_source= os.getenv('DATA_SOURCE_DIR', './source')
data_directory= os.getenv('DATA_DIRECTORY', './data')
log_level = 'INFORMATION'

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def load_data():
    embedding = OpenAIEmbeddings()

    vectordb = Chroma(embedding_function=embedding,persist_directory=data_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents('what are the key deliverables?')
    
    print(docs)

if __name__ == '__main__':
    load_data()

