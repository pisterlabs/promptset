# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:55:33 2023

@author: VermaSum
"""

from flask import Flask, render_template, request
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os
from llama_index import StorageContext, load_index_from_storage

openai_api_key = os.getenv("secretopenai")

app = Flask(__name__)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 1024
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

  
    index.storage_context.persist()

    return index

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['user_input']
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        index = load_index_from_storage(storage_context)
        response = index.as_query_engine()
        response= response.query(input_text)
       

        return render_template('index.html', response=response,user_input=input_text)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    index = construct_index("docs")
    app.run(debug=True)
