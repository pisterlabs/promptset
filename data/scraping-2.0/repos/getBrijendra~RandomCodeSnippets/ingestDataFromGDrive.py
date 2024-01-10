from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.llms import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool

import os
import getpass
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch

# import chromadb
# from chromadb.utils import embedding_functions
from lambda_app.llm.ingestUtils.GoogleDriveLoader import GoogleDriveLoader

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.document_loaders import UnstructuredFileIOLoader

from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate

from langchain.schema.runnable import RunnableMap
from langchain.schema import format_document

from operator import itemgetter
from langchain.memory import ConversationBufferMemory

#from loguru import logger
from langchain.callbacks import FileCallbackHandler
import asyncio
from langchain.callbacks import get_openai_callback

from typing import Tuple, List

import uuid
import os

#logfile = "output.log"

#logger.add(logfile, colorize=True, enqueue=True)
#handler = FileCallbackHandler(logfile)


openai_api_key = ""
openai.api_key = openai_api_key

os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['FOLDER_ID'] = '43f34fwe'

# chroma_db_Client = chromadb.HttpClient(host=os.environ.get('CHROMADB_IP_ADDRESS'), port=8000)
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=openai_api_key,
#                 model_name="text-embedding-ada-002"
#             )

MONGODB_ATLAS_CLUSTER_URI = 'mongodb+srv://abc:abc@brijchatwithdocstest.sfxg1f3.mongodb.net/'
# initialize MongoDB python client
mongo_db_client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
db_name = "langchain_db"
collection_name = "my_collection"
collection = mongo_db_client[db_name][collection_name]
index_name = "langchain_demo"

embeddings = OpenAIEmbeddings()

#Loads Data From GoogleDrive Folders
def loadDataFromGoogleDriveFolder(folder_id: str):
    # folder_id = "1qczk8ORiLNYUNQ3D6h5tCYt70QdmW870"
    folder_id = folder_id
    print(f'FOLDER ID:  {folder_id}')
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        token_path= 'token.json',
        skip_on_failure=True,
        # file_types=["document", "pdf"],
        # file_loader_cls=TextLoader,
        file_loader_kwargs={"mode": "elements"}
    )
    docs = loader.load()
    print(f'Length of the DOCS: {len(docs)}')
    for doc in docs:
        print(doc.metadata)
    return docs

#Loads Data From GoogleDrive Files
def loadDataFromGoogleDriveFiles(file_ids: List[str]):
    print(f'[loadDataFromGoogleDriveFiles] FILE IDs:  {file_ids}')
    loader = GoogleDriveLoader(
        document_ids=file_ids,
        token_path= 'token.json',
        skip_on_failure=True,
        # file_loader_cls=UnstructuredFileIOLoader,
        # file_types=["document", "pdf"],
        # file_loader_cls=TextLoader,
        file_loader_kwargs={"mode": "elements"}
    )
    docs = loader.load()
    print(f'Length of the DOCS: {len(docs)}')
    for doc in docs:
        print("PageContent:", doc.page_content, "\n")
        print("Metadata:", doc.metadata, "\n")
    return docs

#Splits the documents list into Chunks
def textChunker(chunk_size: int, chunk_overlap: int, documents: list):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    return docs

# #Create OpenAI Embeddings and Save It To Chroma
# def createEmbedingsAndSaveToChroma(docs: list):
#     # Set up OpenAI embeddings
#     openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                     api_key=openai_api_key,
#                     model_name="text-embedding-ada-002"
#                 )
#     # load Chroma Client
#     chroma_db_Clients = mongo_db_client
#     # Use 'openai_ef' *OpenAIEmbeddings Function* to create the Collection
#     collection = chroma_db_Clients.get_or_create_collection(name="my_collection", embedding_function=openai_ef)

#     # Save each chunk with the metadata to ChromaDB
#     for doc in docs:
#         # Save Each Document in chromaDb
#         collection.add(
#             ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
#         )


def createEmbedingsAndSaveToChroma(docs: list):
    docsearch = MongoDBAtlasVectorSearch.from_documents(
        docs, embeddings, collection=collection, index_name=index_name
    )



def ingestDataFromGoogleDrive(body: dict):
    folderIds = body.get('folder_ids',[])
    fileIds = body.get('file_ids',[])
    # Ingest Data from folders and push it to Chroma Db
    for folderId in folderIds:
        documents = loadDataFromGoogleDriveFolder(folderId)
        chunkedData = textChunker(500, 100, documents)
        createEmbedingsAndSaveToChroma(chunkedData)
    # Ingest Data from files
    if len(fileIds)!=0:
        documents = loadDataFromGoogleDriveFiles(fileIds)
        chunkedData = textChunker(500, 100, documents)
        createEmbedingsAndSaveToChroma(chunkedData)
    return {"success": True, "message": "Data Ingested Successfully"}, 200

