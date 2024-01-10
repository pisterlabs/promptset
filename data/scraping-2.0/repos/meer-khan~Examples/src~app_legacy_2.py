from flask import Flask,redirect,url_for,request,jsonify, make_response, flash
from markupsafe import escape
from decouple import config
from flask_cors import CORS
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import os
import jsonpickle
from waitress import serve
import pathlib
from datetime import datetime
import requests
import openai
from icecream import ic
import pandas as pd


app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = config("uploadFolder")
cors = CORS(app, resources={r"/prompt/": {"origins": config("ORIGIN")}})
# Get a list of values
# allowed_hosts = config('ALLOWED_HOSTS', default='', cast=lambda v: [s.strip() for s in v.split(',')])

key = config("KEY")
openai.api_key = key

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages


def get_chatgpt_response(messages):
  response = openai.ChatCompletion.create(
  model="gpt-4-1106-preview",
  messages=messages
)
  return  response['choices'][0]['message']['content']


def print_last_message(messages):
    if messages:
        last_message = messages[-1]
        print(last_message["role"] + ": " + last_message["content"])


def get_last_message(messages):
    if messages:
        last_message = messages[-1]
        return last_message["role"] + ": " + last_message["content"]

def initiate_context():
    context = [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "   "},
        ]
    return context

def validator(sc=" ", ccn=" ", seas=" ", ked = " ", iskd= " ", sl= " "): 
    pass

def read_file(file):
    print(file)
    df = pd.read_excel(file)
    # Add file-related information to the chat
    file_info = f"Loaded Excel file '{file}' with {len(df)} rows and {len(df.columns)} columns."
    messages = update_chat(messages, "user", file_info)


@app.route('/prompt/',methods=['POST'])
def pom_extractor():
    try:
        messages = initiate_context()
        data = request.get_json()


        # ques = request.form.get('question')
        # Get the file from the form data
        # file = request.files.get('file')

        # ic(file)

        shop_categories = data.get("shop_categories", "online store")
        content_concept_n_narrative = data.get("ccn", "relevant to shop categories")
        seasionality = data.get("seasonality", "current season")
        key_e_commerce_dates = data.get("ked", "weekly calendar for next 3 months")
        industry_specific_key_days = data.get("iskd", "relevant to shop categories")
        shop_locations = data.get("shop_locations","online store")

        if shop_categories == None: 
            shop_categories = "online store"

        if content_concept_n_narrative == None: 
            content_concept_n_narrative = "relevant to shop categories"

        if seasionality == None: 
            seasionality = "current season"

        if key_e_commerce_dates == None: 
            key_e_commerce_dates = "weekly calendar for next 3 months"

        if industry_specific_key_days == None: 
            industry_specific_key_days = "relevant to shop categories"

        if shop_locations == None: 
            shop_locations = "online store"
        
        ques = f'''
                Please create and rephrase Prompt for weekly calendar campaigns, next 3 months from now, for the shopify store
                {shop_categories} with a focus on {content_concept_n_narrative}
                around {seasionality}, considering {key_e_commerce_dates} and {industry_specific_key_days},
                targeted at {shop_locations}, 
                consider yourself as strategist and implementor and also emphasize to generate data in tabular. 
                I want you to return a table with campaign ideas, reasoning, targeted audiences, Prompt should be 
                tailored according to seasonality, trends, shop category, what the shops products are, key e-commerce days, 
                3 days a week at least. 
            '''
        
        ic (ques)


        messages = update_chat(messages, "user", ques)
        # # ic(messages)
        model_response = get_chatgpt_response(messages)
        # ic(model_response)
        messages = update_chat(messages, "assistant", model_response)
        # ic(messages)
        print_last_message(messages)
        result = get_last_message(messages)
        result = {"response": result}
        result = {"msg": "file uploaded successfully"}
        return make_response(jsonify(result), 200)
    except Exception as e:
        # Handle other exceptions
        error_message = f"An unexpected error occurred: {str(e)}"
        return make_response(jsonify({'error': error_message}), 500)



if __name__ == "__main__":
        
    if config("MODE") == 'DEV':
        app.run(host='localhost', debug=True, port=5000)
    if config("MODE") == 'PROD':
        serve(app, host = '0.0.0.0', port=5000, threads = 4)
        

# max_request_body_size = 2073741824