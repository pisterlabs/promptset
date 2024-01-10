# fast upload (fast-upload.py)

# library imports
import io
import json
import asyncio
import PyPDF2
from fastapi import FastAPI, UploadFile, File
from uvicorn import Server
from sse_starlette import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from langchain.text_splitter import CharacterTextSplitter
import requests
import logging
from logging.handlers import RotatingFileHandler
import sys
import os 
import copy 
from InstructorEmbedding import INSTRUCTOR
from typing import List
import numpy as np
from db_wrapper import VectorDB

# constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler('fast_upload.log', maxBytes=10485760, backupCount=5, encoding='utf-8')   # rotate after 5x10MB log files
stream_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# tell the world we are alive
logger.info("")
logger.info("###################################")
logger.info("")
logger.info("Fast upload service started")
logger.info("")
logger.info("###################################")
logger.info("")

# set environment variables passed in from batch file
QDRANT_URL = os.environ.get("QDRANT_URL")
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH")

logger.info("Loaded environment variables:")
logger.info("EMBEDDING_MODEL_PATH:%s", "[" + EMBEDDING_MODEL_PATH + "]")
logger.info("QDRANT_URL:%s", "[" + QDRANT_URL + "]")

# connect to Vector DB and log collection status
vectorDB = VectorDB(QDRANT_URL, logger)   

# instantiate the embedding LLM
logger.info("Loading model from " + EMBEDDING_MODEL_PATH + "...")
llm = INSTRUCTOR(EMBEDDING_MODEL_PATH)
logger.info("Model loaded")

# create the FastAPI instance
app = FastAPI()    
            
# Set up CORS middleware - add the Access-Control-Allow-Origin header 
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

#define the request body schema
class LLMRequest(BaseModel):
    text: str

# helper function to convert PDF file to text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# helper function to count and log individual embedding
def enumerateEmbedding(embedding):
    for index, number in enumerate(embedding):
        logger.debug("Index:[" + str(index) + "] Number:[" + str(number) + "]")

# retreive embedding for a chunk of text
def get_embedding(chunk):
    embedding = llm.encode(chunk)        
    enumerateEmbedding(embedding)
    embedding_list = np.array(embedding).tolist()
    return(embedding_list)

# helper function to trim the length of a chunk, and remove any newline characters so it prints better
def trimChunk(chunk, max_length):
    chunk = chunk.replace('\n', ' ').replace('\r', ' ')
    if len(chunk) <= max_length:
        return chunk
    else:
        return chunk[:max_length] + "..."

# helper function to trim the length of embedding
def trimEmbedding(embedding,max_length):
    str_repr = ", ".join([str(num) for num in embedding])
    if len(str_repr) <= max_length:
        return str_repr
    else:
        trimmed_str = str_repr[:max_length] 
        return trimmed_str + " ..."
    
# webservice endpoint to load PDF content into vector DB
@app.post("/fast-ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):    
    logger.info("")
    logger.info("/fast-ingest-pdf API called")
    
    # check if file has already been uploaded to the VectorDB
    filename = file.filename
    logger.info("checking if file [" + filename + "] exists in vector database...")        
    if vectorDB.file_exists(filename):
        logger.error("operation aborted, chunks of this file already exist in vector database")
        return {"message" : "operation aborted, chunks of this file already exist in vector database"}
    else:              
        # convert PDF file to text        
        logger.info("confirmed file does not exist in vector database")
        logger.info("converting PDF to text...")
        raw_text = extract_text_from_pdf(io.BytesIO(await file.read()))
        logger.info("converted PDF to text. Text length:" + str(len(raw_text)))        
        
        # split text into chunks  
        logger.info("splitting text into chunks...")
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            length_function = len,
            )

        texts = text_splitter.split_text(raw_text)
        total_chunks = len(texts)
        logger.info("split text into " + str(total_chunks) + " chunks")
        logger.info("")
    
        # process each chunk and persist to Vector db
        i=0
        for chunk in texts:        
            i+=1
            # log chunk processing has started
            logger.info("processing chunk " + str(i) + "/" + str(total_chunks))
            logger.info("chunk size: " + str(len(chunk)))
            logger.info("text:[%s]",trimChunk(chunk, 80))
        
            # call llm to retrieve embedding & log result
            logger.info("retrieving embedding...")
            embedding = get_embedding(chunk)
            logger.info("retrieved embedding vector consisting of " + str(len(embedding)) + " numbers")
            logger.info("embedding:" + "[" + trimEmbedding(embedding, 80) + "]")        
        
            # persist chunk + embedding to vector DB & log result
            logger.info("persisting text chunk and embedding to vector database...")
            vectorDB.persist_chunk(filename, chunk, embedding)                        
                        
        # return result to client    
        logger.info("completed document ingestion")
        return {"result": "PDF file successfully ingested", "text-length" : str(len(raw_text)),"chunks" : str(len(texts)), "chunk size" : CHUNK_SIZE, 
                "chunk_overlap" : str(CHUNK_OVERLAP), "text": raw_text}





