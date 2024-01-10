from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone 
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

import os

os.environ['OPENAI_API_KEY'] = ''
PINECONE_API_KEY = ''
PINECONE_ENV = ''

print("Loading in documents...")

pdf_loader = DirectoryLoader(
    './paper_data/', # created dir
    glob='**/*.pdf', # we only get pdfs
    show_progress=True
)

paper_pdfs = pdf_loader.load()

print("Documents loaded!")

print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=0
)
papers_split = text_splitter.split_documents(paper_pdfs)

print("Documents split!")

# we use the openAI embedding model
embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

print("Creating a pinecone index...")

# Set a name for your index
index_name = 'research-helper'

# Make sure service with the same name does not exist
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)
    
pinecone.create_index(name=index_name, dimension=1536, metric='cosine') 
# dimension is determined according to the openAI embedding model

# Connect to new index
index = pinecone.Index(index_name=index_name)

print("Connected to new Pinecone index!")

print("Uploading data to Pinecone dB...")

papers_db = Pinecone.from_documents(
    papers_split, 
    embeddings, 
    index_name='research-helper'
)

print("Data uploaded to Pinecone dB!")
