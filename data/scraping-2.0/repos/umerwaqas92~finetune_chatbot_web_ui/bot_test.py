from flask import Flask, request, jsonify, render_template, redirect
import requests
import json
import os
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from flask_cors import CORS
import datetime
from PyPDF2 import PdfReader



os.environ["OPENAI_API_KEY"]="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"

def construct_index( api_temp, api_model_name, api_token_max):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 60
    # set chunk size limit
    chunk_size_limit = 600 

    # define LLM
    
    my_dir = os.path.dirname(__file__)
    pickle_file_path = os.path.join(my_dir, 'index.json')
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=api_temp, model_name=api_model_name, max_tokens=api_token_max))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(pickle_file_path).load_data()
    
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index



def ask_ai(query):
    my_dir = os.path.dirname(__file__)
    pickle_file_path = os.path.join(my_dir, 'index.json')
    index2 = GPTSimpleVectorIndex.load_from_disk(pickle_file_path)
    response = index2.query(query, response_mode="compact")
    return {"response": response.response}


# construct_index(api_temp=0.5, api_model_name="text-davinci-003", api_token_max=2000)


print(ask_ai("What is the codecanyone?"))