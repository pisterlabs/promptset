from langchain.document_loaders import PDFPlumberLoader
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

openai_api_key = os.getenv('OPENAI_API_KEY')
promptlayer_api_key = os.getenv('PROMPTLAYER_API_KEY')


# Load documents from the 'docs' directory
documents = []
for file in os.listdir("docs"):
    if file.endswith(".xml"):
        xml_path = "./docs/" + file
        loader = UnstructuredXMLLoader(xml_path)
        documents.extend(loader.load())
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PDFPlumberLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./docs/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())
    elif file.endswith('.csv'):
        text_path = "./docs/" + file
        loader = CSVLoader(text_path)
        documents.extend(loader.load())

# Process and chunk up the text using CharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size = 1000,
   chunk_overlap  = 0
)


documents = text_splitter.split_documents(documents)

# Create the vectorstore using FAISS
vectordb = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())
