import os
from apikey import apikey, serpapikey

import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai

os.environ['OPENAI_API_KEY'] = apikey


# PROMPT
st.title('AMUBHYA AI | EXPERIMENTAL')
prompt = st.text_input('Whatever You Want !')

file_path = "./data/report.pdf"

loader = PyMuPDFLoader(file_path)
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=10)
texts = text_splitter.split_documents(document)
st.write(texts)

persist_directory = "./storage"
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever()
llm = ChatOpenAI()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


if prompt:
        query = prompt
        try:
            llm_response = qa(query)
            st.write(llm_response["result"])
        except Exception as err:
            st.write(str(err))