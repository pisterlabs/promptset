from flask import Flask, render_template, request, current_app, jsonify, session
from flask_session import Session  # 为了使用 Flask session，你需要先安装 Flask-Session
import openai
import logging
from dotenv import load_dotenv

import os

#从配置文件中取出openai的key
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

application = Flask(__name__)
application.config['SECRET_KEY'] = '9a1c2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a'  # 为了使用 session，需要设置一个密钥
application.config['SESSION_TYPE'] = 'filesystem'  # 设置session 存储类型

# Initialize the Session
Session(application)

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/bot', methods=['POST'])
def bot_reply():
    question = request.json['message']
    # Initialize chat history if it does not exist
    if 'chat_history' not in session:
        session['chat_history'] = [{"role": "system", "content": ""}]
    session['chat_history'].append({"role": "user", "content": question})
    response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    temperature=0.0,
    messages = session['chat_history']
    )
    answer = response["choices"][0]["message"]["content"]
    session['chat_history'].append({"role": "assistant", "content": answer})
    #for s in session['chat_history']:
    #    print(s['role'] + ": " + s['content'])                    
    return jsonify({'reply': answer})

if __name__ == '__main__':
        
    application.run(host='0.0.0.0',port=8000)
