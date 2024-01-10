import logging
import sys
import torch
import os
import openai
from llama_index.llms import LlamaCPP
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.utils import messages_to_prompt, completion_to_prompt
from llama_index import ServiceContext
from llama_index import StorageContext
from llama_index.query_engine import CitationQueryEngine
from llama_index import SummaryIndex
from llama_index.llms import OpenAI
from llama_index.indices.loading import load_index_from_storage
import shutil


openai.api_key = open_ai_key



# enviroment variables
# from dotenv import load_dotenv
OPENAI_API_KEY = open_ai_key

llm = LlamaCPP(
    model_path='./upshot-sih-7b-v2.0.gguf',
    temperature = 0.1,
    max_new_tokens = 2000,
    context_window = 8192,
    generate_kwargs = {},
    model_kwargs = {'n_gpu_layers': 40},
    messages_to_prompt = messages_to_prompt,
    completion_to_prompt = completion_to_prompt,
    verbose = True    
)

gllm = OpenAI(temperature= 0.1, model = 'gpt-3.5-turbo-1106')
gservice_context = ServiceContext.from_defaults(llm = gllm , chunk_size= 512)
service_context = ServiceContext.from_defaults(llm = llm , chunk_size= 512)



def prompt(s):
    return f"<|im_start|>system \n you are a Expert legal document Generator. that excels in generating and drafting legal documents.<|im_end|> \n <|im_start|>user \n {s} <|im_end|> \n <|im_start|>assistant \n"

def prompt_simplify(s):
        return f"<|im_start|>system \n you are a AI assistant.that excels in simplify the legal document in simple terms that a non legal person can also understands.<|im_end|> \n <|im_start|> text \n {s} <|im_end|> \n <|im_start|>summary \n"

def prompt_create(s):
        return f"<|im_start|>system \n you are a AI assistant.that excels in generation of the legal document using the attributes provided.<|im_end|> \n <|im_start|> text \n {s} <|im_end|> \n <|im_start|>summary \n"

def prompt_query(s):
        return f"<|im_start|>system \n you are a AI assistant. answer the question in a positive way you have knowledge about the legal data.<|im_end|> \n <|im_start|> text \n {s} <|im_end|> \n <|im_start|>summary \n"

global issummary
global isindex
issummary = False
isindex = False

# api
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware, # https://fastapi.tiangolo.com/tutorial/cors/
    allow_origins=['*'], # wildcard to allow all, more here - https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Origin
    allow_credentials=True, # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Credentials
    allow_methods=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Methods
    allow_headers=['*'], # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Allow-Headers
)

@app.post("/generate")
async def generate_text(s : dict):
    s = s['s']
    return llm.complete(prompt(s)).text.replace('[/INST]', '').strip()

@app.post("/simplify_text")
async def simplify_text(s : dict):
    s = s['s']
    return llm.complete(prompt_simplify(s)).text.replace('[/INST]', '').strip()

@app.post("/form_request")
async def form_request(s : dict):
    if len(s['s']) > 1:
        query = 'create a legal draft for ' + s['s']['selectedForm'] + " with the following attributes:"
        for i, (key, value) in enumerate(s['s'].items()):
            if i >= 1:
                query += key + " : " + value  + "\n"
    else:
        query = 'create a legal draft for ' + s['s']['selectedForm']

    return llm.complete(prompt_create(query)).text.replace('[/INST]', '').strip()


@app.post("/search")
def query_model(s : dict):
    query = s['s']
    index_storage_context = StorageContext.from_defaults(persist_dir='index')
    index = load_index_from_storage(service_context = service_context , storage_context = index_storage_context) 
    #query_engine = CitationQueryEngine.from_args(index, similarity_top_k = 2, citation_chunk_size= 512)
    query_engine = index.as_query_engine()
    summary_storage_context = StorageContext.from_defaults(persist_dir='summary_index')
    summary_index = load_index_from_storage(service_context = gservice_context , storage_context = summary_storage_context)
    summary_query_engine = summary_index.as_query_engine(
         response = 'tree_summarize',
         use_async = True
    )
    if 'summary' in query or 'summarize' in query or 'simplify' in query:
        return summary_query_engine.query(query).response
    else:
         return query_engine.query(prompt(query)).response
    
     

@app.post('/upload_file')
async def upload_files(file: UploadFile = File(...)):
    global issummary
    global isindex
    file_name = file.filename
    with open(f"data/upload.pdf", "wb") as f:
        f.write(await file.read())

    documents = SimpleDirectoryReader('data/').load_data()

    if isindex == True and issummary == True:
        if os.path.exists('index/'):
            shutil.rmtree('index/')
        if os.path.exists('summary_index/'):
            shutil.rmtree('summary_index/')
        isindex = False 
        issummary = False
    index = VectorStoreIndex.from_documents(documents = documents, service_context= service_context)
    summary_index = SummaryIndex.from_documents(documents = documents, service_context= gservice_context)

    index.storage_context.persist('index')
    summary_index.storage_context.persist('summary_index')
    isindex = True
    issummary = True
    os.remove(f"data/upload.pdf")

    return {'status' : 'success'}


#uvicorn upshot_backend:app --host 0.0.0.0 --port 80 --reload

