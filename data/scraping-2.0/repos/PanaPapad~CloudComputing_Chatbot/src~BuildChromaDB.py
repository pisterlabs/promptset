import os
import re
from langchain.vectorstores import Chroma  # Importing Chroma vector stores
from langchain.document_loaders import TextLoader  # Importing TextLoader for documents
from langchain.embeddings.openai import OpenAIEmbeddings  # Importing OpenAI Embeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter  # Importing text splitters
from langchain.docstore.document import Document  # Importing Document from langchain
from langchain.document_transformers import Html2TextTransformer  # Importing Html2TextTransformer
from urllib.parse import urljoin  # Importing urljoin from urllib.parse
import requests  # Importing requests library for handling HTTP requests
from bs4 import BeautifulSoup  # Importing BeautifulSoup for HTML parsing
from langchain.document_loaders import AsyncHtmlLoader  # Importing AsyncHtmlLoader for loading HTML content asynchronously

# URL to scrape
urls = "https://ucy-linc-lab.github.io/fogify/"

# Function to check if the script is running inside a Docker container
def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )

# Function to clean HTML tags from raw HTML content
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')  # Regular expression to identify HTML tags
    cleantext = re.sub(cleanr, '', raw_html)  # Removing HTML tags
    return cleantext

# Setting up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Initializing OpenAI embeddings and Chroma vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma("langchain_store", embeddings, persist_directory="./data/CHROMA_DB_STABLE")
vectorstore.persist()  # Persisting the vector store

# Setting the directory for documents based on Docker presence
if is_docker():
    docs_dir = "./data/documentation"
else: 
    docs_dir = "./documentation"

# Initializing RecursiveCharacterTextSplitter for text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300, separators=[" ", ",", "\n"])

# Initializing Html2TextTransformer for transforming HTML to text
html2text = Html2TextTransformer()

# Looping through files in the specified directory and its subdirectories
for root, dirs, files in os.walk(docs_dir):
    for file in files:
        print(file)
        if file.endswith('.html') :
            # Loading HTML document and transforming it to text
            raw_document = TextLoader(os.path.join(root, file)).load()
            documents = text_splitter.split_documents(raw_document)
            documents = html2text.transform_documents(documents)
            vectorstore.add_documents(documents)  # Adding documents to the vector store

# Persisting the updated vector store
vectorstore.persist()

# Performing similarity search in the vector store
print(vectorstore.similarity_search("What is Fogify?"))
