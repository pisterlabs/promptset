import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os
from docx import Document
from youtube_transcript_api import YouTubeTranscriptApi as yta
import re
import requests
import re
import pickle
from bs4 import BeautifulSoup
import csv

load_dotenv()
openai_api_key=os.getenv("OPEN_AI_API_KEY")
conversation=""
def pdf__data(pdffile):
    text = ""
    #Read pdffile then stock everything in text 
    for pdf in pdffile:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def docs_data(docfile):
    text = ""
    doc = Document(docfile)
    
    for paragraph in doc.paragraphs:
        text = text + os.linesep + paragraph.text

    return text


def tube__data(tube_link):
    link=tube_link.split("v=")[1]
    text_mkhrbq= yta.get_transcript(link)
    text=""
    for key,value in text_mkhrbq.items():
        if(key=='text'):
            text+=value
    return text


def webb__data(web_link):
    link_request = requests.get(web_link)

    all_content= BeautifulSoup(link_request.content, "html.parser")
    text=all_content.get_text()
    return(text)


def csv_data(csvv):
    with open(csvv, newline='') as csvfile:
        csvvv = csv.reader(csvfile)
        text = '\n'.join(','.join(row) for row in csvvv)
        return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print("All good in chunks")
    return chunks


def get_vectorstore(text_chunks, directory_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("All good in vectores")
    with open(directory_path+"/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
def load_vectorstore(directory_path):
    with open(directory_path+"/vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print("All good in Chain")
    return conversation_chain


def handle_userinput(user_question,conversation):
    response = conversation({'question': user_question})
    print(response)
    chat_history = response['chat_history']
    print(chat_history)



def process_file(file_path):
    # ndiro variables li ghadi ikon fihom text dyal each file
    result=""

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        result=pdf__data(file_path)
    elif file_extension == '.docx':
        result=docs_data(file_path)
    elif file_extension == '.csv':
        result=csv_data(file_path)
    elif file_extension == '.txt':
        with open(file_path, 'r') as file:
            file_content = file.read()
            if 'youtube' in file_content.lower():
                result=tube__data(file_content)
            else:
                result=webb__data(file_content)
    else:
        print(f"Skipping {file_path} - unsupported file type")
    return result
    

