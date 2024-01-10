from fastapi import FastAPI
import logging
import uvicorn
import streamlit as st
from components.sidebar import sidebar
import pinecone
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from loguru import logger
import sys
import json
from fastapi import Response
from pydantic import BaseModel
import os


# set to DEBUG for more verbose logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key  = os.getenv("OPENAI_API_KEY")
# s3 = S3("classgpt")

def init_logger():
    logger.add("295API.log", level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {message} | {extra[user]}", rotation="10 MB", compression="gz")


#client = OpenAI(api_key="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")
client = OpenAI(api_key=openai_api_key)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.get("/topic")
async def topic(query: str = None):
    print(query)
    index_id = "295-youtube-index"
    pinecone.init(
        api_key=pinecone_api_key,
        environment="us-west1-gcp-free"
    )
    pinecone_index = pinecone.Index(index_id)
    # encoded_query = client.embeddings.create(input=query,  model="text-embedding-ada-002")['data'][0]['embedding']
    # res = query_gpt(chosen_class, chosen_pdf, query)
    text = query.replace("\n", " ")
    encoded_query = client.embeddings.create(
        input=[text], model="text-embedding-ada-002").data[0].embedding
    response = pinecone_index.query(encoded_query, top_k=3,
                                    include_metadata=True)
    elements = []
    # Create json from list
    url = []
    for m in response['matches']:
        url.append(m['metadata']['url'])
    logger.bind(user="1").info(f"Topic: {query} |")
    return Response(content=json.dumps(url), media_type="application/json")


@app.get("/question")
async def question(query: str = None):
    index_id = "295-youtube-index"
    pinecone.init(
                api_key=pinecone_api_key,
                environment="us-west1-gcp-free"
                )
    pinecone_index = pinecone.Index(index_id)
    pinecone_index.describe_index_stats()
            
    # Encode the query using the 'text-embedding-ada-002' model
    #encoded_query = client.embeddings.create(input=query,model="text-embedding-ada-002")['data'][0]['embedding']
    encoded_query = client.embeddings.create(input = [query], model="text-embedding-ada-002").data[0].embedding
    response = pinecone_index.query(encoded_query, top_k=20,
                            include_metadata=True)
    context = ""
    for m in response['matches']:
        context += "\n" + m['metadata']['text']
            
    url = ""
    st.subheader("References")
    for m in response['matches'][0:2]:
        url += m['metadata']['url']   
    prompt=f"Please provide a concise answer in markdown format tothe following question: {query} based on content below{context} and the internet. If you are not sure, then say I donot know."
    query_gpt = [
        {"role": "system", "content": "You are a helpful teachingassistant for computer organization"},
        {"role": "user", "content": f"""{prompt}"""},
        ]
    answer_response = client.chat.completions.create(model='gpt-3.5-turbo',
        messages= query_gpt,
        temperature=0,
        max_tokens=500)
    answer = answer_response.choices[0].message.content
    
    
    # Create json from list
    url = []
    for m in response['matches']:
        url.append(m['metadata']['url'])
    logger.bind(user="1").info(f"Topic: {query} |")
    
    response_json = {"answer": answer, "references": url}
    
    return Response(content=json.dumps(response_json), media_type="application/json")



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=42000,
        log_level="debug",
        ssl_keyfile="/repo/key.pem",  # Path to your key file
        ssl_certfile="/repo/cert.pem"  # Path to your certificate file
    )
