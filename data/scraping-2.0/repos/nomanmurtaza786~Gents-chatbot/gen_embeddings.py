
import os

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from supabase_db import saveDocVectorSupabase, saveToSupabase

embeddingOpenAi= OpenAIEmbeddings()

# loader = PyPDFLoader('/Users/nomanmurtaza/Documents/Noman_Murtaza_CV.pdf')
loader = CSVLoader(file_path="/Users/nomanmurtaza/Downloads/user details.csv")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
pages = loader.load_and_split(text_splitter=text_splitter)

#saveDocVectorSupabase(pages)