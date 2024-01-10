from src.utils import load_pdf,text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#loading pdf
extracted_data = load_pdf("data/")

#store chunks
text_chunks = text_split(extracted_data)

#Downloading Embeddings
embeddings = download_hugging_face_embeddings()

#initializing Pinecone
pinecone.init(      
	api_key='c3701bf9-f5f8-4b3d-82da-118a22b743bf',      
	environment='gcp-starter')      
index_name = 'medibot'

#Creating Embeddings for each of the text chunks and sorting
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
