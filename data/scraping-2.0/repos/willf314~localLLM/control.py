# AI Control Service (control.py)

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

from db_wrapper import VectorDB

# constants

EMBED_VECTOR_SIZE = 5120
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler('control.log', maxBytes=10485760, backupCount=5, encoding='utf-8')   # rotate after 5x10MB log files
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
logger.info("AI control service started")
logger.info("")
logger.info("###################################")
logger.info("")

# set environment variables passed in from batch file
LLM_QUERY_URL = os.environ.get("LLM_QUERY_URL")
LLM_QUERY_ASYNC_URL = os.environ.get("LLM_QUERY_ASYNC_URL")
LLM_EMBEDDING_URL = os.environ.get("LLM_EMBEDDING_URL")
QDRANT_URL = os.environ.get("QDRANT_URL")

logger.info("Loaded environment variables:")
logger.info("LLM_QUERY_URL:%s", "[" + LLM_QUERY_URL + "]")
logger.info("LLM_QUERY_ASYNC_URL:%s", "[" + LLM_QUERY_ASYNC_URL + "]")
logger.info("LLM_EMBEDDING_URL:%s", "[" + LLM_EMBEDDING_URL + "]")
logger.info("QDRANT_URL:%s", "[" + QDRANT_URL + "]")

# connect to Vector DB and log collection status
vectorDB = VectorDB(QDRANT_URL, logger)   

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

# retreive embedding for a chunk of text
def get_embedding(chunk):

    # Define the endpoint URL
    endpoint_url = LLM_EMBEDDING_URL

    # Define the request payload
    payload = {
        "text": chunk
    }

    # Make a POST request to the LLM service
    response = requests.post(endpoint_url, json=payload)

    # Check the response status code
    if response.status_code == 200:
        # Get the response JSON
        response_json = response.json()
        embedding = response_json["embedding"]
        # Process the embedding as needed        
        return(embedding)
    else:
        logger.error("Request failed with status code: %d", response.status_code)
        return()

def query_llm(prompt):
    
    # define the endpoint URL
    endpoint_url = LLM_QUERY_URL

    # define the request payload
    payload = {
        "text": prompt
    }

    # make a POST request to the LLM service
    response = requests.post(endpoint_url, json=payload)

    # check the response status code
    if response.status_code == 200:
        # Get the response JSON
        response_json = response.json()
        
        # return response       
        return(response_json)
    else:
        logger.error("Request failed with status code: %d", response.status_code)
        return()

def query_llm_async(prompt):
    
    # define the endpoint URL
    endpoint_url = LLM_QUERY_ASYNC_URL

    # define the request payload
    payload = {
        "text": prompt
    }

    # make a POST request to the LLM service
    response = requests.post(endpoint_url, json=payload)

    # check the response status code
    if response.status_code == 200:                        
        # return response       
        return("async query successful")
    else:
        logger.error("Request failed with status code: %d", response.status_code)
        return("async query failed")

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
@app.post("/ingest-pdf")
async def ingest_pdf(file: UploadFile = File(...)):    
    logger.info("")
    logger.info("/ingest-pdf API called")
    
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

# web service API to query docs
@app.post("/query-docs")
async def query(request: LLMRequest):
    
    logger.info("/query-docs API called")
    logger.info("Query:[%s]", request.text)
    logger.info("")

    # get embedding for query from llm
    logger.info("retrieving embedding for query...")
    embedding = get_embedding(request.text)
    logger.info("retrieved embedding vector consisting of " + str(len(embedding)) + " numbers")
    logger.info("embedding:" + "[" + trimEmbedding(embedding, 80) + "]")    

    # retrieve matching chunks from vector DB
    logger.info("retrieving matching chunks from Vector DB using a similarity search...")
    search_results=vectorDB.retrieve_matches(embedding)
    
    # todo - only continue if search found results

    # create prompt for LLM - join query with the context information retrieved from the Vector DB
    context = vectorDB.convert_to_string(search_results)
    prompt = "Using the provided context, answer the question. Assume everything in the context is true. "
    prompt += "Context:" + context + " "            
    prompt += "Question:" + request.text + " "
    prompt += "Answer:"
        
    logger.debug("created prompt with context information:[%s]", prompt)
    
    #call LLM
    logger.info("calling llm-query to get answer...")
    result = query_llm(prompt)
    
    #retrieve answer + supporting data from the LLM result            
    answer = result["answer"]
    id = result["id"]
    object = result["object"]
    created = result["created"]
    modelName = result["modelName"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    total_tokens = result["total_tokens"]

    #log results 
    logger.info("question:[%s]",request.text)
    logger.info("prompt:[%s]",prompt)
    logger.info("answer:[%s]",answer)        
    logger.info("id:[%s]", str(id))
    logger.info("object:[%s]", object)
    logger.info("created:[%s]", created)    
    logger.info("model:[%s]", modelName)
    logger.info("prompt_tokens:[%s]", str(prompt_tokens))
    logger.info("completion_tokens:[%s]", str(prompt_tokens))
    logger.info("total_tokens:[%s]", str(total_tokens))
    
    #return result to client
    return {"question" : request.text, "prompt" : prompt , "answer" : answer, "id" : id, "object" : object,
           "created" : created, "modelName" : modelName, 
           "prompt_tokens" : prompt_tokens, "completion_tokens" : completion_tokens,
           "total_tokens" : total_tokens}


# web service API to query docs and trigger response via event stream
# todo -refactor to reduce duplication of code between sync & async versions of the function

@app.post("/query-docs-async")
async def query(request: LLMRequest):
    
    logger.info("/query-docs-async API called")
    logger.info("Query:[%s]", request.text)
    logger.info("")

    # get embedding for query from llm
    logger.info("retrieving embedding for query...")
    embedding = get_embedding(request.text)
    logger.info("retrieved embedding vector consisting of " + str(len(embedding)) + " numbers")
    logger.info("embedding:" + "[" + trimEmbedding(embedding, 80) + "]")    

    # retrieve matching chunks from vector DB
    logger.info("retrieving matching chunks from Vector DB using a similarity search...")
    search_results=vectorDB.retrieve_matches(embedding)    

    # todo - only continue if search found results

    # create prompt for LLM - join query with the context information retrieved from the Vector DB
    context = vectorDB.convert_to_string(search_results)
    prompt = "Using the provided context, answer the question. Assume everything in the context is true. "
    prompt += "Context:" + context + " "            
    prompt += "Question:" + request.text + " "
    prompt += "Answer:"
        
    logger.debug("created prompt with context information:[%s]", prompt)
    
    #call LLM
    logger.info("calling llm-query to get answer...")
    result = query_llm_async(prompt)
    
    #retrieve answer + supporting data from the LLM result            
    logger.info(result)
        
    #return result to client
    return {"message" : result}




