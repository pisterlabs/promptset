from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"]= st.secrets['OPENAI_API_KEY']

class User_Input(BaseModel):
    query: str

app= FastAPI()

with st.sidebar:
    st.title('fiXit PDF Chatbot')
    st.markdown('''
                ## About
                App developed using OpenAI API, Langchain, and streamlit
                ''')

def make_api_call(query):
    base_url= "https://localhost:8000/getinput"
    response= requests.post(base_url, json= {'query': query})
    return response.json()

def compute(query):
    docs= VectorStore.similarity_search(query= query, k= 3)
    llm= OpenAI()
    chain= load_qa_chain(llm= llm, chain_type= "stuff")
    with get_openai_callback() as cb:
        response= chain.run(input_documents= docs, question= query)
        #print(cb)
    st.write(response)

@app.post("/getinput")
async def trigger(user_input: User_Input):
    compute(user_input.query)
    return {"message":"Prediction Completed"}

st.header("Chat with your PDF")
pdf= st.file_uploader("Upload your PDF", type= 'pdf')

if pdf is not None:
    pdf_reader= PdfReader(pdf)

    text= ""
    for page in pdf_reader.pages:
        text+= page.extract_text()

    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap= 200,
        length_function= len
    )
    chunks= text_splitter.split_text(text= text)

    pdf_name= pdf.name[:-4]
    st.write(f'{pdf_name}')

    if os.path.exists(f"{pdf_name}.pkl"):
        with open(f"{pdf_name}.pkl", "rb") as f:
            VectorStore= pickle.load(f)
    else:
        embeddings= OpenAIEmbeddings()
        VectorStore= FAISS.from_texts(chunks, embedding= embeddings)
        with open(f"{pdf_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

    query= st.text_input("Ask questions about your PDF file:")
    if query:
        response= compute(query)
