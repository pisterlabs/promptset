from openai import OpenAI
import httpx
from conf.constants import *
from langchain.prompts import PromptTemplate

from qdrant_client import QdrantClient
from qdrant_client.http import models
import glob
import traceback

from multiprocess import Process, Queue
import time
import queue # imported for using queue.Empty exception

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import sys
import regex as re

import argparse
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---

# create an embedding using openai
@retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(1))
def get_embedding(openai_client, text, model="text-embedding-ada-002"):
   start = time.time()
   text = text.replace("\n", " ")
   resp = openai_client.embeddings.create(input = [text], model=model)
   print("Embedding ms: ", time.time() - start)
   return resp.data[0].embedding

PROMPT_TEMPLATE = PromptTemplate.from_template(
        """
        What are the top 20 entities mentioned in the given context? 
        Extract any part of the context AS IS that is relevant to answer the question. 
        At the end, provide a brief summary about the whole context.
        
        > Context:
        >>>
        {text}
        >>>

        Exclude these entities in your response:
        - Apache Camel
        - Java 
        - Maven 
        - Red Hat

        """
    )

# extract keywords using the chat API with custom prompt
@retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(1))
def extract_keywords(openai_client, document):
    start = time.time()
    message = PROMPT_TEMPLATE.format(text=document)
    response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a service used to extract entities from text"},
                {"role": "user", "content": message}
            ]
        )
    print("Extraction ms: ", time.time() - start)
    return response.choices[0].message.content

def create_openai_client():
    client = OpenAI(
        timeout=httpx.Timeout(
            10.0, read=8.0, write=3.0, connect=3.0
            )
    )
    return client

def create_qdrant_client(): 
    client = QdrantClient(
       QDRANT_URL,
        api_key=QDRANT_KEY,
    )
    return client
        
# --- 

# arguments
parser = argparse.ArgumentParser(description='Upsert PDF pages')
parser.add_argument('-c', '--collection', help='The target collection name', required=True)
parser.add_argument('-s', '--start', help='Start of the batch', required=False, default=0)
parser.add_argument('-b', '--batchsize', help='Batch size (How many pages)', required=False, default=10)
parser.add_argument('-p', '--processes', help='Number of parallel processes', required=False, default=2)
parser.add_argument('-m', '--mode', help='Parser mode (pdf|web)', required=False, default="pdf")
parser.add_argument('-f', '--file', help='Upsert indivual file', required=False)
args = parser.parse_args()

# the regex used to extract a reference form the filename
ID_REF_REGEX = "\/([^\/]+)$" # defaults to PDF mode
if(args.mode == "web"):
    ID_REF_REGEX = "_([^_]+)_([^_]+)$"

filenames = []
if(args.file is None):
    for _file in glob.glob(TEXT_DIR+args.collection+"/*.txt"):
        filenames.append(_file)
else:
    filenames.append(args.file)

# sort
def pagenum(name):
    match = re.search("_([^_]+).txt$", name)[1]    
    return int(match)

if(args.mode == "pdf"):
    filenames.sort(key=pagenum)
else:
    filenames.sort()

# preparations for ingestion
docfiles = []
start = int(args.start)
end = int(args.start)+int(args.batchsize)

# guardrails
if end >= len(filenames):
    end = len(filenames)

for name in filenames[start:end]:
    
    file_content = None
    with open(name) as f:                
        file_content = f.read()
    
    page_ref = re.search(ID_REF_REGEX, name)[0]
    
    # split files if needed
    threshold = 2500
    if(len(file_content)> threshold):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = threshold,
            chunk_overlap  = 100            
        )
        chunks = text_splitter.split_text(file_content)

        for i,chunk in enumerate(chunks):
            docfiles.append({
                "page": str(page_ref)+"_"+str(i),
                "content": chunk
            })        
    else:
        docfiles.append({
            "page": str(page_ref),
            "content": file_content
        })

print("Upserting N pages: ", len(docfiles))

# start with a fresh DB everytime this file is run from a zero index
if(start==0 and args.file is None):
    print("Recreate collection ", args.collection)
    create_qdrant_client().recreate_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(
            size=1536,  # Vector size is defined by OpenAI model
            distance=models.Distance.COSINE,
        ),
    )
else:
    print("Upsert into exisitng collection ", args.collection)

def do_job(tasks_to_accomplish):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''

            page_ref = str(task["page_ref"])
            page_content = task["page_content"]
            
            print("Start page '"+ page_ref+ "'")
            
            try:
                                
                openai_client = create_openai_client()    

                # extract keywords                                
                entities = extract_keywords(openai_client, page_content)
                
                # create embeddings                          
                embeddings = get_embedding(openai_client, text=entities)                

            except Exception as e:
                print("Failed to call openai (skipping ... ): ", page_ref)                
                print(e)
                continue            

            try:    

                qdrant_client = create_qdrant_client()
            
                # Upsert        
                upsert_resp = qdrant_client.upsert(
                    collection_name=args.collection,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embeddings,
                            payload={
                                "page_content": "\""+page_content+"\"",
                                "metadata": {
                                    "page_number": page_ref,
                                    "entities": entities            
                                }
                            }
                        )
                    ]        
                )
               
                
                
            except Exception as e:
                print("Failed to upsert page (skipping ... ): ", page_ref)
                print(e)
                continue            

            print("Page ", page_ref, " completed \n")
            
    return True


def main():
    
    number_of_processes = int(args.processes)
    tasks_to_accomplish = Queue()
    
    processes = []

    for doc in docfiles:
        tasks_to_accomplish.put(
            {
                "page_ref": doc["page"],
                "page_content": doc["content"]
            }
        )

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=[tasks_to_accomplish])
        processes.append(p)
        p.start()

    # completing processes
    for p in processes:
        p.join()
    
    return True


if __name__ == '__main__':
    main()
