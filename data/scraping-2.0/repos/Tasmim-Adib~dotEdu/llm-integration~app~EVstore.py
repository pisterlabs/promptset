import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv


import pinecone

# from KBsearch import *

load_dotenv()

os.environ["PINECONE_API_KEY"] = os.environ.get('pinecone_apikey')
os.environ["PINECONE_ENV"] = os.environ.get('pinecode_env')
os.environ["OPENAI_API_KEY"] = os.environ.get('openAI_apikey')


embeddings = OpenAIEmbeddings()


def initializePinecone():
    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

# First, check if our index already exists. If it doesn't, we create it
def createIndex(index_name:str):
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  
    )


def newDataLoad(directory:str):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        separators=[" ", ",", "\n"]
    )
    return text_splitter.split_documents(documents)



def newIndex(index_name:str, directory:str):
    docs = newDataLoad(directory=directory)
    return Pinecone.from_documents(docs, embeddings, index_name=index_name)

def existingVectorstore(index_name:str):
    # if you already have an index, you can load it like this
    return Pinecone.from_existing_index(index_name, embeddings)

def addToIndex(index_name:str, documents:list):
    index = pinecone.Index(index_name)
    text_field = "text"
    vectorstore = Pinecone(index, embeddings, text_field)
    vectorstore.add_documents(documents)



def extractDocMetaData(docs:list):
    content = [x.page_content for x in (docs)]
    metadata = [x.metadata for x in (docs)]
    return content, metadata

def testing():
    vectorstore = existingVectorstore("kazinazrul")
    query = "কত সালে ব্রিটিশ ভারতীয় সেনাবাহিনীতে যোগ দিয়েছিলেন তিনি?"
    docs = vectorstore.similarity_search(query, k=2)
    # print(docs)


# text_field = "text"
# index = pinecone.Index("dotedu")
# vectorstore = Pinecone(index, embeddings, text_field)

# print(result)


# index = pinecone.GRPCIndex(index_name)
# print(index.describe_index_stats())
