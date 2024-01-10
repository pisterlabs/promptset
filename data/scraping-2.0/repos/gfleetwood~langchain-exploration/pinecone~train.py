from io import StringIO
import sys
from typing import Dict, Optional

#from langchain.agents import load_tools
#from langchain.agents import initialize_agent
#from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader, UnstructuredPDFLoader, OnlinePDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
import pinecone
from os import environ

pinecone.init(api_key = environ['PINECONE_API_KEY'], environment = environ['PINECONE_API_ENV'])

llm = OpenAI(temperature = 0.0)

loader = UnstructuredHTMLLoader("../data/asd.html")
data = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key = environ['OPENAI_API_KEY'])

index_name = "INDEX"
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name = index_name)
