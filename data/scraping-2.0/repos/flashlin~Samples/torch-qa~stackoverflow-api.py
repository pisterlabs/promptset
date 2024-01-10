from langchain.llms import CTransformers
from langchain.chains import LLMChain, RetrievalQA
from langchain import PromptTemplate
import streamlit as st
import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from typing import Dict
import uvicorn

MODEL_FILE = "models/stablecode-instruct-alpha-3b.ggmlv1.q5_1.bin"
MODEL_TYPE = "gpt_neox"
MODEL_FILE = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"
MODEL_TYPE = "llama"

app = FastAPI()

app.mount("/static", StaticFiles(directory="data-StackOverflow"), name="public")

templates = Jinja2Templates(directory="templates")
DB_FAISS_PATH = 'models/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


def load_llm():
    llm = CTransformers(
        model=MODEL_FILE,
        model_type=MODEL_TYPE,
        max_new_tokens=512,
        temperature=0.7
    )
    return llm


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='microsoft/unixcoder-base',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_response")
async def get_response(request: Request, query: str = Form(...)):
    resp = final_result(query)
    result = resp['result']
    print(resp)
    for i in resp['source_documents'][0]:
        if 'metadata' in i:
            source_doc = i[1]['source']
    # print(resp['source_documents'][0]['metadata']['source'])
    response_data = jsonable_encoder(json.dumps({"result": result, "source_doc": source_doc}))
    res = Response(response_data)
    return res


if __name__ == "__main__":
    uvicorn.run("stackoverflow-api:app", host='0.0.0.0', port=8000, reload=True)
