#OS imports
import os
import json
from datetime import datetime

#File/shell tools
import shutil

import langchain

#Bring in the utils
from utils import init_logging, init_db

#Bring in the layout parser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

#PDF tools
import PyPDF2

#Env Mgt
from dotenv import load_dotenv

api_key = os.getenv('OPENAI_API_KEY')
data_conn = os.getenv('DATA_CONNECTION_STRING', './data/database.db')
data_processed= os.getenv('DATA_PROCESSED_DIR', './source/processed')
data_pre_processed= os.getenv('DATA_PRE_PROCESSED_DIR', './source/preprocessed')
data_source= os.getenv('DATA_SOURCE_DIR', './source')
data_directory= os.getenv('DATA_DIRECTORY', './data')
log_level = 'INFORMATION'

# Initialize logging
logger = init_logging(log_level)

# Initialize database connection
conn = init_db(data_conn)

loader = PyPDFLoader(f'{data_source}/NAP NPP Final with print outs 09.06.16.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory=data_directory
)
vectordb.persist()
