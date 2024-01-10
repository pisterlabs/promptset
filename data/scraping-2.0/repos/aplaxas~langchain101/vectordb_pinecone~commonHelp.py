from dotenv import load_dotenv
import os 
import pinecone
import pinecone.info

from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

def getPinecone():
  load_dotenv()

  pk = os.getenv("PINECONE_API_KEY")
  penv = os.getenv("PINECONE_ENV")

  pinecone.init(api_key=pk, environment=penv)

  index_name = "docqa"
  index = pinecone.Index(index_name) 
  index_stats_response = index.describe_index_stats()

  print(index_stats_response)

  return index

def getEmbeddings():
    load_dotenv()
    openaikey = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=openaikey, model="text-embedding-ada-002")
    return embeddings


def split_docs(file, chunk_size=1000, chunk_overlap=100):
    print('Start splitting docs')
    loader = UnstructuredPDFLoader(file)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(document)
    print('End splitting docs')
    return doc

def embedding(folder, file, indexName, embedding, tag):
     print("Start embedding " + file)
     docs = split_docs(folder + file)
     for doc in docs:
      doc.metadata = {"source": file, "Tag": tag}

     Pinecone.from_documents(docs, embedding, index_name=indexName)
     print("End embedding " + file)


