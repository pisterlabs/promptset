from fastapi.responses import JSONResponse,StreamingResponse
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import uuid
import os
from chat_bot import *
from pydantic import BaseModel

api_key = os.environ['openai_api_key']
embedding=OpenAIEmbeddings(openai_api_key=api_key)
openai.api_key = api_key
collection_name = 'simple'
persist_dir = './chromadb'
def init_db():
    vectorstore = Chroma(collection_name=collection_name,persist_directory=persist_dir,embedding_function=embedding)
    return vectorstore
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Rule(BaseModel) : 
    rule : str


def add_rule(text):
    Chroma.from_texts(collection_name=collection_name,texts=[text], embedding=embedding ,persist_directory=persist_dir)


def compare_similarity(query):
    vectorstore = init_db()
    result = vectorstore.similarity_search_with_relevance_scores(query, k=5)
    score_list = []
    for i in result:
        score_list.append(i[-1])
    try:
        average = sum(score_list)/len(score_list)
        return average
    except:
        return 0

def clear_db_collection():
    vectorstore = init_db()
    res = vectorstore.delete_collection()

def get_collection():
    vectorstore = init_db()
    result = vectorstore.get()
    return result['documents']

def generate_response(res):
    for i in res:
        yield i
    
@app.post('/create_rule')
def create_rule(rule:Rule):
    add_rule(rule.rule)
    return JSONResponse(content={
        "message":"success",
        "rule":rule.rule
        })
@app.get('/clear_collection')
def clear_collection():
    clear_db_collection()
    return JSONResponse(content={"message":"remove complete"})

@app.get('/rule')
def get_rule():
    res = get_collection()
    return JSONResponse(content={"result":res})

@app.get("/session")
def session():
    client_uuid = uuid.uuid4()
    create_session(str(client_uuid))
    result = {
        "uuid" : str(client_uuid)
    }
    return JSONResponse(content=result)

@app.get("/query")
async def main(uuid:str,message:str):
    get_session(uuid)
    collection = get_collection()
    if  len(collection) == 0:
        return StreamingResponse(
                    generate_response('No rules configure please ask admin'), 
                    media_type="text/event-stream"
                )
    else:
        score = compare_similarity(message)
        if score > 0.7:
            return StreamingResponse(
                        stream_chat(
                            uuid = uuid,
                            prompt= message
                        ), 
                        media_type="text/event-stream"
                    )
        else:
            return StreamingResponse(
                        generate_response('bad prompt please ask another one'), 
                        media_type="text/event-stream"
                    )
