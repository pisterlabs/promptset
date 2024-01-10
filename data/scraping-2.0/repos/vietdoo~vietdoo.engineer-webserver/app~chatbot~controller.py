from app import app
from flask import render_template, request, Blueprint, jsonify , after_this_request
from config import *
import os
import openai
import json
from dotenv import load_dotenv
load_dotenv()

import time
import datetime
from app import conn
chatbot_page = Blueprint('chatbot', __name__, url_prefix='/chatbot')

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_cors import CORS, cross_origin
import mysql.connector

token = GPT_TOKEN
openai.api_key = token



print("token:", token)

def generate_response(prompt):
    result = openai.ChatCompletion.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature = 0.9
    )

    #print("waiting response from OPEN AI.")
    
    message = result.choices[0].message['content']
    message = message.strip('\n')
    
    return message


@chatbot_page.route('/')
def home():
    return render_template('chatbot/index.html')

@chatbot_page.route('/dev')
def homedev():
    return render_template('chatbot/index.html')

@chatbot_page.route('/text')
def text():
    return "Chào Fen !"


@cross_origin()
@chatbot_page.route("/", methods=['GET', 'POST'])
def chatbot():
    if request.headers.getlist("X-Forwarded-For"):
        clientIP = request.headers.getlist("X-Forwarded-For")[0]
    else:
        clientIP = request.remote_addr
    #clientIP = request.environ['REMOTE_ADDR']
    #clientPORT = request.environ['REMOTE_PORT']
    print(f"Client IP: {clientIP}")
    resTime = time.time()
    data = request.get_json()
    prompt = data["prompt"]
    print(f"Text: {prompt}")
    message = generate_response(prompt)
    resTime = time.time() - resTime
    #message = "Mời bạn quay lại sau nhé, Bơ đang uống sữa🧂"
    print(f"Reponse: {message}")
    
    try:
        cursor = conn.cursor()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(timestamp)
        sql = f"""INSERT INTO 
                chatLog(id, clientIP, request, response, resTime, code, date) 
                VALUES(default, '{clientIP}', '{prompt}', '{message}', {resTime}, 200, '{timestamp}');
        """
        try:
            cursor.execute(sql)
            conn.commit()

        except Exception as e:
            print(e)
            conn.rollback()
    except Exception as e:
        print(e)
        pass
    return message



@chatbot_page.route('/test')
def test():
    print('testing route: on')
    return generate_response("chào bạn")