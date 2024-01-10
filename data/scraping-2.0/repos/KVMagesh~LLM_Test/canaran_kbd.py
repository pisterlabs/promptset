# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:12:46 2023

@author: Mahesh
"""

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma 
#from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
#import os
'''
folder = ''
#folder = './pdf'
sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
for folder in sub_folders:
    print('going to folder --->',folder)
    directory = os.path.join(os.getcwd(), r'file/',folder)
    #directory = os.path.join(os.getcwd(), r'pdf',folder)
  
    if len(os.listdir(directory)) == 0:
        print("Empty directory :  ",directory)
    else:
        print("Not empty directory :",directory)
        loader = PyPDFDirectoryLoader(directory)
        docs = loader.load()
        print(docs)
        collection_name=folder
        persist_directory="./Knowledge Base/"+collection_name
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectordb = Chroma.from_documents(documents, embedding=embeddings,collection_name=collection_name, 
                                      persist_directory=persist_directory)
        vectordb.persist()
'''     

directory='./file'
loader = PyPDFDirectoryLoader(directory)
documents = loader.load()
collection_name='canara_kdb'
persist_directory="./canara/kdb/"+collection_name
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
print('text_splitter:',text_splitter)
docs = text_splitter.split_documents(documents)
print('docs:',docs)
hf_embedding = HuggingFaceInstructEmbeddings()
print('embedding:',hf_embedding)
vectordb = Chroma.from_documents(docs, embedding=hf_embedding,collection_name=collection_name, 
                                      persist_directory=persist_directory)
vectordb.persist()
print('done')
query = "what is Accidental Death Benefit?"
print(query)
search = vectordb.similarity_search(query, k=2)
print('search result:',search)