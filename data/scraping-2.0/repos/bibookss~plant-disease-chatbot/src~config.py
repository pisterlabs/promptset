from langchain.chat_models import ChatOllama
from langchain.document_loaders import WebBaseLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler

import os

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def initalize_model(stream_handler):
    ollama = ChatOllama(
        base_url='http://localhost:11434', 
        model="orca-mini", 
        temperature=0.7,
        streaming=True,
        callbacks=[stream_handler],
        num_gpu=7,
        num_thread=8
        )

    return ollama

def initialize_db():
    # Load the documents from links.txt
    links_dir = os.path.join(os.getcwd(), 'data', 'links.txt')
    links = open(links_dir, 'r').read().split('\n')

    # Create the vectorstore
    documents = []
    # for link in links:
    #     print(f"Loading {link}")
    #     data = WebBaseLoader(link).load()
    #     documents.extend(data)

    # Get the html files from the data directory
    dir = os.path.join(os.getcwd(), 'data')
    files = os.listdir(dir)
    files = [os.path.join(dir, file) for file in files if file.endswith('.html')]
    for file in files:
        print(f"Loading {file}")
        data = UnstructuredHTMLLoader(file).load()
        documents.extend(data)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(documents)

    db = Chroma.from_documents(documents=chunked_documents, embedding=GPT4AllEmbeddings())
    db.persist()

    return db