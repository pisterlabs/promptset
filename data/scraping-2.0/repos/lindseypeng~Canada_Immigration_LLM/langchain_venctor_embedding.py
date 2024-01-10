from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

import pandas as pd
import openai
import os 
pd.options.display.max_colwidth = None

openai_api_key = os.environ.get('OPENAI_API_KEY')
print(len(openai_api_key))

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="ImmigrationCanada_Sponsorship_QA_may7.csv")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

persist_directory = 'docs/chroma/'

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings,persist_directory=persist_directory)
print(vectordb._collection.count())
vectordb.persist()


