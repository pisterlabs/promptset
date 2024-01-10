import os 
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

file_name = 'books/Romeo_and_Juliet.txt'
openai_api_key = os.environ['OPENAI_KEY']

raw_text = TextLoader(file_name).load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_text)
db = Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory='embeddings'+ file_name)
