import modal
import openai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import urlparse
import os
from hashlib import md5
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from configs import *

documents_mt = modal.Mount.from_local_dir(local_path=DOCUMENTS_DIR, remote_path='/')
embeddings_fs = modal.NetworkFileSystem.persisted('tailgate-embeddings')

image = (
    modal.Image.from_registry('python:3.11-slim-bookworm', setup_dockerfile_commands=["RUN apt-get update", "RUN apt-get install build-essential -y"])
    .pip_install_from_requirements("requirements.txt")
    .copy_mount(documents_mt, remote_path='/root/documents')
)

stub = modal.Stub("tailgate-api", image=image)
web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_key_origin(request):
    key = request.headers.get("x-api-key")
    domain = urlparse(request.headers.get("origin")).hostname.replace('www.', '')
    if key not in DOMAIN_WHITELIST:
        return False
    if domain not in DOMAIN_WHITELIST[key]:
        print("Invalid origin: " + domain)
        return False
    return True

@web_app.post("/")
def f(req: Request):
    if not check_key_origin(req):
        return {"error": "Invalid origin"}

    return {"Hello": "World"}

@stub.function(secret=modal.Secret.from_name("openai-secret"), network_file_systems={'/root/embeddings': embeddings_fs})
def create_or_replace_vectorstore():
    files = [f for f in os.listdir(DOCUMENTS_DIR) if f != '.DS_Store']
    file_with_sizes = {}
    for f in files:
        file_with_sizes[f] = os.path.getsize(DOCUMENTS_DIR + '/' + f)
    sorted_files = sorted(file_with_sizes.items(), key=lambda x: x[1], reverse=True)
    vs_hash = md5(str(sorted_files).encode('utf-8')).hexdigest()
    vecstore_path = EMBEDDINGS_DIR + '/' + vs_hash

    if os.path.exists(vecstore_path):
        return vecstore_path
    else:
        loaders = [PDFMinerLoader(DOCUMENTS_DIR + '/' + f) for f in files if f.endswith('.pdf')] # TODO: add more loaders
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splits = splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=vecstore_path)
        vectorstore.persist()

        return vecstore_path

@stub.function(secret=modal.Secret.from_name("openai-secret"), network_file_systems={'/root/embeddings': embeddings_fs})
def query_vectorstore(prompt: str):
    vecstore_path = create_or_replace_vectorstore.remote()
    vectorstore = Chroma(persist_directory=vecstore_path, embedding_function=OpenAIEmbeddings())

    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
    result = qa_chain({"query": prompt})
    response = result['result']

    docs_used = []
    if len(result['source_documents']) > 0:
        docs_used = list(set(documents.metadata['source'] for documents in result['source_documents']))

    return response, docs_used

class AskDocsRequest(BaseModel):
    prompt: str

@web_app.post("/ask-docs")
def ask_docs(request: AskDocsRequest, req: Request):
    prompt = request.prompt
    if not check_key_origin(req):
        return {"error": "Invalid origin"}

    response, docs_used = query_vectorstore.call(prompt)
    return {"response": response, "docs_used": docs_used}

class GenerateTextRequest(BaseModel):
    prompt: str

@web_app.post("/generate-text")
def generate_text(request: GenerateTextRequest, req: Request):
    prompt = request.prompt
    if not check_key_origin(req):
        return {"error": "Invalid origin"}

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    
    return response['choices'][0].message['content']

class GenerateImageRequest(BaseModel):
    prompt: str

@web_app.post("/generate-image")
def generate_image(request: GenerateImageRequest, req: Request):
    prompt = request.prompt
    if not check_key_origin(req):
        return {"error": "Invalid origin"}

    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=IMAGE_SIZE
    )

    image_url = response['data'][0]['url']
    return image_url

@stub.function(secret=modal.Secret.from_name("openai-secret"))
@modal.asgi_app()
def app():
    return web_app

if __name__ == "__main__":
    stub.serve()