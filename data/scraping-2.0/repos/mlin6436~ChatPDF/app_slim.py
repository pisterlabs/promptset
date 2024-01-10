import dotenv
import os
import streamlit as st
import tempfile
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.header("Chat PDF")

file = st.file_uploader("Upload a PDF file", type=["pdf"])
if file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file.getbuffer())
        file_path = tf.name
    doc = PyPDFLoader(file_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    knowledge_base = FAISS.from_documents(docs, embeddings)

    question = st.text_input("Ask your question here:")
    if question:
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=knowledge_base.as_retriever())
        response = chain.run(question)
        
        st.success("Completed question.")
        st.write("Answer: ", response)