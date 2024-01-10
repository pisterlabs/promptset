from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader


app = FastAPI()

class Request(BaseModel):
    text: str

def chat(query):
    
    pdf_reader = PdfReader(r'D:\UMBC ANIL\Fall 2023\DATA 690 NLP\Chatbot project\Project\UMBC_International.pdf')
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    #print(text)

    # split into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500,
                                      chunk_overlap=10, length_function=len)

    text_chunks = char_text_splitter.split_text(text)

    api_key = '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'
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    docsearch = FAISS.from_texts(text_chunks, embeddings)

    llm = OpenAI(openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")

    query  = query
    docs = docsearch.similarity_search(query )
    response = chain.run(input_documents=docs, question=query)
    print(" ")
    print(query)
    print(response)
    return response


@app.post("/")
async def home(request: Request):
    try:
        response_text= chat(request.text)
        if response_text:           
            if response_text =="I don't know":
                request.text="Please contact UMBC for further assistance."
            res = {"fulfillment_response": {"messages": [{"text": {"text": [response_text]}}]}}
            return res

        raise Exception("No fulfillment text found in request")

    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": "Invalid request format"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
