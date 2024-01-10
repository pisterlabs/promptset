import os
from apikey import apikey, serpapikey
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import nltk

os.environ['OPENAI_API_KEY'] = apikey

# PROMPT
st.title('AMUBHYA AI | EXPERIMENTAL')
prompt = st.text_input('Whatever You Want !')


loader = DirectoryLoader('./data/', glob='**/*.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)
llm = OpenAI()
qa = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=docsearch.as_retriever(),
                                return_source_documents=True)

if prompt:
    response = qa({"query": prompt})
    st.write(response['result'])
    st.write(response['source_documents'])