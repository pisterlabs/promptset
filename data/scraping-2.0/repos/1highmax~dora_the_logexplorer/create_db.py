# Import libraries
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
import re
import hashlib

def extract_log_data(line):
    # Regular expression to match the time, prefix, and message
    pattern = r'(\b\w{3}\s\d{2}\s\d{2}:\d{2}:\d{2}\b)\s+(.*?)\s*:\s*(.*)'
    match = re.match(pattern, line)
    if match:
        time, prefix, message = match.groups()
        return time, prefix, message
    else:
        return None, None, None
    



def create_database(file_path):
    print("starting")
    # OpenAI embeddings
    embedding = OpenAIEmbeddings()
    log_data=[]
    lines = []
    file_path = file_path
    persist_directory = 'db'

    loader = TextLoader(file_path, encoding = 'UTF-8')
    doc = loader.load()
    len(doc)

    # Splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200) #1000,200
    texts = text_splitter.split_documents(doc)

    # Count the number of chunks
    len(texts)
        
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line)
            time, prefix, message = extract_log_data(line)
            log_data.append({'time': time, 'prefix': prefix, 'message': message})


    # Initialize the vector database with the unique persist directory
    vectordb = Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    vectordb.persist()

    print("Vectordb initialized with unique persist directory:", persist_directory)