from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import GooglePaLMEmbeddings

from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

client = MongoClient(host='mdburl')
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]
loader = PyPDFLoader("/Users/akhilkumarp/development/personal/github/LLM_Journey/RagOverCode/AnthropicLatest3.pdf", extract_images=True)
pages = loader.load()
pages[4].page_content
print("Yes - 1")
