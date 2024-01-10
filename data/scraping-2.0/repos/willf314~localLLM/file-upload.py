# file upload (file-upload.py)

# library imports
import io
import json
import asyncio
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import sys
import os 
import copy 
from InstructorEmbedding import INSTRUCTOR
from typing import List
import numpy as np
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo, ScoredPoint, Record

# vector db constants
COLLECTION_NAME = "collection1"
MAX_CHUNKS_TO_RETURN = 3
TRIM_CHUNK_LEN = 80   

# Vector database wrapper class 
class VectorDB:

    # constructor for VectorDB wrapper class. Connects to the specified DB service endpoint
    def __init__(self, db_url):
        self.url = db_url
        
        # connect to Qdrant vector database
        print("connecting to qdrant db at [" + self.url + "] ...")
        self.vectorDB = QdrantClient(self.url)
        print("succesfully connected to vector DB")

        # show collection status info. this will throw an error if collection has not been created with initVectorDB 
        print("connecting to collection [" + COLLECTION_NAME + "]...")
        collectionInfo = self.vectorDB.get_collection(collection_name=COLLECTION_NAME) 
        print("collection status:")
        print("   status:", str(collectionInfo.status))
        print("   points_count:", str(collectionInfo.points_count))
        print("   vectors_count:", str(collectionInfo.vectors_count))
        print("   segments_count:", str(collectionInfo.segments_count))    
        print("   payload_schema:", str(collectionInfo.payload_schema))
    
            
    # persist chunk to vector db - Qdrant
    def persist_chunk(self, source_filename, text_chunk, embedding):  
        
        # create a uuid based on hostname and time
        idx = str(uuid.uuid1())
        
        # insert chunk into vector db
        self.vectorDB.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=idx, 
                vector=embedding,
                payload={"text": text_chunk, "source_filename" : source_filename}
            )        
        ]
        )

        # log summary info
        print("persisted chunk to db:")
        print("  id:[" + idx + "]")
        print("  chunk size:" + str(len(text_chunk)))
        print("  source_filename:[" + source_filename + "]")
        
        return()

    # returns true if the file already exists in the vectorDB
    def file_exists(self, source_filename):
        # look for any existing chunks with this filename. Note we only need 1 result
        search_result = self.vectorDB.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_filename",
                        match=models.MatchValue(value=source_filename),
                    ),
                ]
            ),
        )

        # unpack search_result. Could find nothing in API doco on purpose of the Union object 
        records, union_obj = search_result
        
        # count results - count property didn't seem to work?
        
        i=0
        for record in records:
            i += 1
            print("oops - found chunk with matching source_filename:")
            print("  id:[" + str(record.id) + "]")
            print("  payload:[" + trimChunk(str(record.payload), TRIM_CHUNK_LEN) + "]")
                                                    
        # return true if we found at least 1 matching chunk         
        return ( i > 0 )

    # join search result payload together to form a context string    
    def convert_to_string(self, search_results):
        # for now we take the full Qdrant payload JSON including field names and values and concatenate it into a string
        context=""
        for result in search_results:
            context += json.dumps(result.payload) + " "   
        return context


# helper function to convert PDF file to text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# retreive embedding for a chunk of text
def get_embedding(chunk):
    embedding = llm.encode(chunk)        
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
              
# helper function to trim the length of a chunk, and remove any newline characters for logging purposes
def trimChunk(chunk, max_length):
    chunk = chunk.replace('\n', ' ').replace('\r', ' ')
    if len(chunk) <= max_length:
        return chunk
    else:
        return chunk[:max_length] + "..."

# helper function to count and log individual embedding
def enumerateEmbedding(embedding):
    for index, number in enumerate(embedding):
        print("Index:[" + str(index) + "] Number:[" + str(number) + "]")

# main program

print("starting file-upload.py", flush = True)

# constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# set environment variables 
QDRANT_URL = os.environ.get("QDRANT_URL")
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH")

print("Loaded environment variables:")
print("EMBEDDING_MODEL_PATH:[" + EMBEDDING_MODEL_PATH + "]")
print("QDRANT_URL:[" + QDRANT_URL + "]")

# connect to Vector DB 
vectorDB = VectorDB(QDRANT_URL)   

# instantiate the embedding LLM
print("Loading model from " + EMBEDDING_MODEL_PATH + "...")
llm = INSTRUCTOR(EMBEDDING_MODEL_PATH)
print("Model loaded")
    
# Retrieve the file name from command line arguments
if len(sys.argv) < 2:
    print("Please provide a file name as a command line argument.")
    sys.exit(1)
file_path = sys.argv[1]
# file_path = os.path.join(os.getcwd(), file_arg)

# open file
print("ingesting file: file_path [" + file_path + "]")
with open(file_path, 'rb') as file:
    # check if file has already been uploaded to the VectorDB
    filename = file.name
    print("checking if file [" + filename + "] exists in vector database...")        
    if vectorDB.file_exists(filename):
        print("operation aborted, chunks of this file already exist in vector database")        
    else:              
        # convert PDF file to text        
        print("confirmed file does not exist in vector database")
        print("converting PDF to text...")
        raw_text = extract_text_from_pdf(file)
        print("converted PDF to text. Text length:" + str(len(raw_text)))        
        
        # split text into chunks  
        print("splitting text into chunks...")
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            length_function = len,
            )

        texts = text_splitter.split_text(raw_text)
        total_chunks = len(texts)
        print("split text into " + str(total_chunks) + " chunks")
            
        # process each chunk and persist to Vector db
        i=0
        for chunk in texts:        
            i+=1
            # log chunk processing has started
            print("processing chunk " + str(i) + "/" + str(total_chunks))
            print("chunk size: " + str(len(chunk)))
            print("text:",trimChunk(chunk, 80))
        
            # call llm to retrieve embedding & log result
            print("retrieving embedding...")
            embedding = get_embedding(chunk)
            print("retrieved embedding vector consisting of " + str(len(embedding)) + " numbers")
            print("embedding:" + "[" + trimEmbedding(embedding, 80) + "]")        
        
            # persist chunk + embedding to vector DB & log result
            print("persisting text chunk and embedding to vector database...")
            vectorDB.persist_chunk(filename, chunk, embedding)                        
                                  
        print("completed document ingestion")


