# PDF Loaders. If unstructured gives you a hard time, try PyPDFLoader
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

loader = UnstructuredPDFLoader("./tesla/Tesla_Model_3.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)


# print(texts[1])

OPENAI_API_KEY = "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"
PINECONE_API_KEY = "d4e2ee4e-2f42-47b9-bd6b-2b3b48ac393e"
PINECONE_API_ENV = "asia-southeast1-gcp-free"
PINECONE_INDEX_NAME = "tesla"


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "tesla"  # put in the name of your pinecone index here

docsearch = Pinecone.from_texts(
    [t.page_content for t in texts], embeddings, index_name=index_name)

query = "What is the overall weight of Model3?"
docs = docsearch.similarity_search(query)

print(docs)
