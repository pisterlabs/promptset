from os import getenv
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import pipeline, AutoTokenizer
from torch import cuda
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig #must build from source and install. git clone https://github.com/PanQiWei/AutoGPTQ && pip install .

app = FastAPI()

#This is unsafe. please change to production URL when deploying into prod
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_dotenv()  # take environment variables from .env.

    global model, tokenizer, instructorEmbeddings, db, generate_text
    model_path = getenv("MODEL_PATH")
    cache_folder = getenv("CACHE_FOLDER")
    model_base_name= getenv("MODEL_BASE_NAME")
    instructor_model_name = getenv("INSTRUCTOR_MODEL_NAME")
    milvus_host = getenv("MILVUS_HOST")
    milvus_port = getenv("MILVUS_PORT")
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    model = AutoGPTQForCausalLM.from_quantized(model_path, use_safetensors=True, model_basename=model_base_name, device=device, use_triton=False, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, cache_folder=cache_folder)

    generate_text = pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        device=device,
        temperature=0.0,
        top_p=0.15,
        top_k=0,
        max_new_tokens=1000,
        repetition_penalty=1.1 
    )

    print(generate_text("Tell me the difference between Kubernetes and Docker."))

    instructorEmbeddings = HuggingFaceInstructEmbeddings(model_name=instructor_model_name, model_kwargs={"device": "cuda"}, cache_folder=cache_folder)
    db = Milvus(instructorEmbeddings, connection_args={"host": milvus_host, "port": milvus_port})


@app.get("/")
async def readyResponse():
    return {"Status": "Ready"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"/app/{file.filename}"
    with open(file_location, "wb+") as f:
        f.write(await file.read())
    
    pdfLoader = PDFMinerLoader(file_path=file_location)
    loadedPdfDocuments = pdfLoader.load()

    rcc_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    split_pdf = rcc_text_splitter.split_documents(loadedPdfDocuments)
    milvus_host = getenv("MILVUS_HOST")
    milvus_port = getenv("MILVUS_PORT")
    db.from_documents(split_pdf, embedding=instructorEmbeddings, connection_args={"host": milvus_host, "port": milvus_port})
    return {"filename": file.filename + "Has been successfully uploaded and processed."}

@app.post("/query")
async def query_files(query: str):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=HuggingFacePipeline(pipeline=generate_text), chain_type="stuff", retriever=retriever)
    
    response = qa_chain(query)
    return {"response": response}

@app.post("/testquery/{query}")
async def test_query(query: str):
    return {"response": generate_text(query)}