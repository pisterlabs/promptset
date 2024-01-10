import os
from dotenv import load_dotenv

load_dotenv()

from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

TARGET_FILE = "data/Team () - EPAM AI Hackathon Group Connect_2023-12-13.docx"

loader = Docx2txtLoader(TARGET_FILE)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", " ", "\t", "\r", "\f", "\v"], chunk_size=500, chunk_overlap=200,
)

docs = text_splitter.create_documents([doc[0].page_content])

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

embeddings = OpenAIEmbeddings()
db = Pinecone.from_documents(docs, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))
