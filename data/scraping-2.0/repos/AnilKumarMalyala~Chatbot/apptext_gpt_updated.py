#importing necessary libraries

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
#from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from textblob import TextBlob


app = FastAPI()

class Request(BaseModel):
    text: str

import time

request_data = []
last_request_time = None

#function to set timer for 30 seconds(to overcome OpenAI limitation)
def make_request_time(data):
    global last_request_time, request_data
    print(f'\nLast Request Time is {last_request_time}.')
    current_time = time.time()
    print(f'Current Request Time is {current_time}.\n')
    if last_request_time is not None:
        time_elapsed = current_time - last_request_time
        if time_elapsed < 30:
            request_data.append(data)
            time_to_wait = 30 - time_elapsed
            time.sleep(time_to_wait)
            data_to_send = request_data.copy()
            request_data.clear()
            last_request_time = time.time()
        else:
            request_data.append(data)
            last_request_time = time.time()
    else:
        request_data.append(data)
        last_request_time = time.time()
    print(f'Updated last request time is {last_request_time}.\n')

#function to correct spelling errors in query if any.
def correct_spelling(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

#function to tackle unkown queries
def checking_response(response):
    if "i don't know" in response.lower():
        return "The information you need doesn't appear to be in the International arrival guide. If you believe this inquiry is valid, please email to internationalquery@umbc.edu. One of our student counselors will repond to your query with 2-3 business days."
    return response

#query function
def chat(query):
    query = correct_spelling(query)
    make_request_time(query)
    pdf_reader = PdfReader(r'D:\UMBC ANIL\Fall 2023\DATA 690 NLP\Chatbot project\Project\UMBC_International.pdf') #text corpus
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # splitting the text into chunks
    char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500,
                                               chunk_overlap=10, length_function=len)

    text_chunks = char_text_splitter.split_text(text)

    api_key = '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'
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    docsearch = FAISS.from_texts(text_chunks, embeddings)

    llm = OpenAI(openai_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    print(f'The Query is: {query}\n')
    print(f'The Response generated is: {response}\n')
    return checking_response(response)


@app.post("/")
async def home(request: Request):
    try:
        response_text= chat(request.text)
        if response_text:           
            res = {"fulfillment_response": {"messages": [{"text": {"text": [response_text]}}]}}
            return res
        raise Exception("No fulfillment text found in request")

    except Exception as e:
        print(str(e))
        return JSONResponse(content={"error": "Invalid request format"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
