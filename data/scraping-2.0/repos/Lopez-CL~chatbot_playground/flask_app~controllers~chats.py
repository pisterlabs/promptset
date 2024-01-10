from flask_app import app
from flask import request, render_template, redirect, session, flash, jsonify, Response
from flask_session import Session 
import os, requests
from dotenv import load_dotenv, find_dotenv
import openai
# Render page with chatbot interface
@app.route('/')
def render_chatbot():
    session['messages'] = [
    {"role": "system", "content": "You are a helpful assistant."},
    ]
    print(session['messages'])
    return render_template('index.html')
# My prompt completion method
openai.api_key = os.getenv('OPENAI_API_KEY')
def get_completion(context, model='gpt-3.5-turbo', temperature=0):
    response = openai.ChatCompletion.create(model=model, messages=context,temperature=temperature)
    return response.choices[0].message['content']

@app.route('/get/completion', methods = ['post'])
def join_completion_with_messages():
    user_message = request.form['user-prompt']
    if 'messages' in session:
        session['messages'].append({'role': 'user','content': user_message})
        response = get_completion(session['messages'])
        session['messages'].append({'role': 'assistant', 'content': response})
        print(session['messages'])
        result = session['messages'][-1:]
        print(result[0]['content'].split())
        return jsonify(result[0]['content'].split())

@app.route('/clear/session', methods=['Get'])
def clear_chat_session():
    session.clear()
    session['messages'] = [
    {"role": "system", "content": "You are a helpful assistant."},
    ]
    print(session['messages'])
    return Response(status = 204)