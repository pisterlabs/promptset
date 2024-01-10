#app.py
from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_login import login_user, logout_user, current_user, login_required#, LoginManager
from flask_bcrypt import Bcrypt

import os
from datetime import timezone
import datetime
from app import app, login_manager
import app.chat_db as db
from app.models import User
from app.conversation import Conversation as conv
import openai

if not db.check_db_exists():
    db.build_db()

#from dotenv import load_dotenv
#load_dotenv()

# Check if using Azure or OpenAI API 
api_type = os.environ.get('OPENAI_API_TYPE')
if api_type == 'AZURE':
    openai.api_type = api_type
    openai.api_version = os.environ.get('OPENAI_API_VERSION')
    openai.api_base = os.environ.get('OPENAI_API_BASE')
    deployment = os.environ.get('OPENAI_DEPLOYMENT')
else:
    deployment = None

openai.api_key = os.environ.get('OPENAI_API_KEY')

system_prompt = "You are an AI assistant that talks like a pirate in rhyming couplets."

def get_timestamp():
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    return round(utc_time.timestamp())
    
@app.route('/')
@app.route('/index')
@login_required
def home():
    try:
        template = render_template('index.html')
        return template
    except Exception as e:
        print(e)
        return "not found"
    
@app.route("/login", methods = ['POST','GET'])
def login():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            username = username.lower()
            try:
                remember_me = request.form['remember_me']
            except:
                remember_me = False

            if remember_me == 'true':
                remember = True
            else:
                remember = False
            user = User.get_user(username)
            # dummy password check for now
            if user and User.check_password(password,user.password_hash):
                login_user(user, remember=remember)
                print('user valid')
                template = redirect(url_for('home'))
            else:
                error = 'Invalid username or password.'
                template = render_template('login.html', message=error)
        except Exception as e:
            print(e)
            template = render_template('login.html', message=e)
        finally:
            return template
    else:
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        template = render_template('login.html')
        return template

@app.route("/register", methods = ['POST','GET'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            username = username.lower()
            created = User.create_user(username,password)
            if created:
                template = redirect(url_for('login'))
            else:
                #template = render_template('register.html', message = 'user not created')
                raise Exception(created)
        except Exception as e:
            print(e)
            template = render_template('register.html', message=e)
        finally:
            return template
    else:
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        template = render_template('register.html')
        return template

@app.route("/message",methods=["POST"])
@login_required
def message():
    try:
        conversation_id = request.form['conversation_id']
        content = request.form['message']
        conversation_id = int(conversation_id)
        # flag -1 for new conversation create conversation and add system prompt
        if conversation_id == -1:
            user = current_user
            convo = conv.new_conversation(user.username)
            conversation_id = convo.id
            convo = conv.get_conversation(conversation_id)
            convo.add_message('system',system_prompt)
        # for continued conversations get convo from db
        else:
            convo = conv.get_conversation(conversation_id)
        # add new message
        convo.add_message('user',content)
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=convo.messages,
            engine=deployment
        )
        response = chat['choices'][0]['message']['content']
        convo.add_message('assistant',response)
        data = {"id":conversation_id,"response":response}
        return jsonify(data)
    except Exception as e:
        print(e)
        return e

@app.route("/history",methods=["GET"])
@login_required
def history():
    try:
        user = current_user
        data = conv.get_history(user.username)
        return jsonify(data)
    except Exception as e:
        print(e)
        return e

@app.route("/get_conversation",methods=["POST"])
@login_required
def get_conversation():
    try:
        conversation_id = request.form['conversation_id']
        conversation_id = int(conversation_id)
        convo = conv.get_conversation(conversation_id)
        data = []
        for m in convo.messages[1:]:
            data.append(m['content'])
        return jsonify(data)
    except Exception as e:
        print(e)
        return e