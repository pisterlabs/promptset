from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

loader = DirectoryLoader('data', glob="**/*.txt")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="./vectorstore")



