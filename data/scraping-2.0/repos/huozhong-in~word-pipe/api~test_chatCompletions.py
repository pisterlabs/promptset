#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import time
import hashlib
import random
from pathlib import Path
import marisa_trie
from flask import (
    Flask, 
    Response, 
    make_response, 
    jsonify, 
    request, 
    render_template, 
    url_for, 
    send_file,
    g,
    stream_with_context)
from flask_sse import sse
from stardict import *
from config import *
from io import BytesIO
from PIL import Image
from user import User,UserDB
from chat_record import ChatRecord,ChatRecordDB
from flask_cors import CORS
import requests
import openai
import logging
import streamtologger
from Crypto.Cipher import AES
import base64


app = Flask(__name__)
CORS(app)


@app.route('/api/openai/v1/chat/completions', methods=['POST'])
def openai_chat():
    '''
    代理OpenAI Chat API，供flutter的openai包调用。支持stream
    '''
    apiKey = request.headers.get('Authorization').split(' ')[1] if request.headers.get('Authorization') != None else None
    openai.api_key = apiKey
    openai.proxy = PROXIES['https']
    model = request.json['model']
    messages = request.json['messages']
    stream: bool = False
    if request.json.get('stream') != None and request.json.get('stream')==True: # and request.headers.get('Accept') == 'text/event-stream':
        stream = True
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=stream
    )

    def generate():
        start_time = time.time()
        for chunk in response:
            chunk_time = time.time() - start_time
            print(f"Message received {chunk_time:.2f} seconds after request:")
            yield json.dumps(chunk) + '\n'
        
    if stream:
        headers: dict = {}
        headers['Accept'] = 'text/event-stream'
        headers['X-Accel-Buffering'] = 'no'
        headers['Cache-Control'] = 'no-cache, must-revalidate'
        headers['Connection'] = 'keep-alive'
        return Response(stream_with_context(generate()), headers=headers, content_type='application/json; charset=utf-8')
    else:
        return make_response(jsonify(response), 200)