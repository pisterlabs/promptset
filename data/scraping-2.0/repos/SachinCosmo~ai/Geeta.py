import os
import getpass
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# load huggingface api key
os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]

# use streamlit file uploader to ask user for file
# file = st.file_uploader("Upload PDF")


path = "https://vedpuran.files.wordpress.com/2013/04/455_gita_roman.pdf"
loader = PyPDFLoader(path)
pages = loader.load()

# st.write(pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=os.environ["HUGGINGFACE_HUB_TOKEN"], model_kwargs={'temperature': 0.2,'max_length': 1000})

from langchain.schema import retriever
retireval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=doc_search.as_retriever())

if query := st.chat_input("Enter a question: "):
  with st.chat_message("assistant"):
    st.write(retireval_chain.run(query))
