import os
import sys

from apikey import apikey

import streamlit as st

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


os.environ["OPENAI_API_KEY"] = apikey

loader= TextLoader('data.txt')

st.title("AgriQNet_GPT")
prompt = st.text_input("Your Questions here: ")

llm = OpenAI(temperature= 0.9)

if prompt:
    response= llm(prompt)
    st.write(response)