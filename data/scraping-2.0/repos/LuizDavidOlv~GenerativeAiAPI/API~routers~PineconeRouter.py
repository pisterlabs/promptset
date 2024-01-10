import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pinecone
from API.Models.PineconeCreateIndexModel import CreateIndexModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

class Instructions(BaseModel):
    instruction: str

router = APIRouter(
    prefix="/pinecone",
    tags=["Pinecone"]
)

@router.get("/version/")
def version():
    return pinecone.info.version()

@router.get("/list-indexes/")
def list_indexes():
    return pinecone.list_indexes()

@router.post("/describe-index/")
def describe_index(index_name: str):
    return pinecone.describe_index(index_name)

@router.post("/create-index/")
def create_index(data: CreateIndexModel):
    return pinecone.create_index(data.index_name, dimension= data.dimension, metric = data.metric, pods= 1, pod_type='p1.x2')
    
@router.post("/upsert-index/")
def upsert_index(index_name: str, text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_documents(chunks,embeddings,index_name=index_name)

@router.delete("/delete-index")
def delete_index(index_name: str):
    return pinecone.delete_index(index_name)
    


