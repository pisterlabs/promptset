# This file will contain the code for ingesting the document and putting it upto pinecone

import os 
from langchain.document_loaders import ReadTheDocsLoader,PyPDFLoader,DataFrameLoader # It will basically take the documnetations and convert it into the lagchain based docs.
# Read the docs loader is useful for documentation reading.
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings 

pinecone.init(api_key="YOUR API KEY HERE",environment="asia-southeast1-gcp-free") 


def ingest_doc():
    loader =PyPDFLoader(r'C:\Users\karan\Test\japji_sahib.pdf')
    raw_doc=loader.load()
    splitted_doc = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100,separators=["\n\n",'\n',' ',''])
    print(splitted_doc)
    raw_doc= splitted_doc.split_documents(raw_doc)  
    print(raw_doc[0])  
#     embeddings= OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))  
# ## Right Now I am not preprocessing the data as for learning purposes, but later on I would like to preprocess the data so that I can 
# # create something of my own.
#     Pinecone.from_documents(raw_doc,embeddings,index_name='doc-chat')# currently revoked index_name # this will basically takes the chunks of the data and make use if openai embeddings to

ingest_doc() 

# We Have successfully transferred the vectors in the pinecone 
 