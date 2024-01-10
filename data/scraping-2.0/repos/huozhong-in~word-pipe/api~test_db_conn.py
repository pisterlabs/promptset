#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import time
import hashlib
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
)
from flask_sse import sse
from stardict import *
from config import *
from io import BytesIO
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import scoped_session, sessionmaker
from user import User,UserDB
from chat_record import *
from flask_cors import CORS
import requests
import openai
import logging
from Crypto.Cipher import AES
import base64
from queue import Queue
import tiktoken

app = Flask(__name__)
CORS(app)

# logging
if not os.path.exists('logs'):
    os.mkdir('logs')
logging.basicConfig(
    filename='logs/api_{starttime}.log'.format(starttime=time.strftime('%Y%m%d', time.localtime(time.time()))),
    filemode='a',
    level=logging.DEBUG,
    format='%(levelname)s:%(asctime)s:%(message)s'
)
stderr_logger = logging.getLogger('STDERR')
# import streamtologger
# streamtologger.redirect(target="logs/print.log", append=False, header_format="[{timestamp:%Y-%m-%d %H:%M:%S} - {level:5}] ")

# for SSE nginx configuration https://serverfault.com/questions/801628/for-server-sent-events-sse-what-nginx-proxy-configuration-is-appropriate
app.config["REDIS_URL"] = REDIS_URL
@sse.after_request
def add_header(response):
    response.headers['X-Accel-Buffering'] = 'no'
    # response.headers['Cache-Control'] = 'no-cache'
    # response.headers['Connection'] = 'keep-alive'
    response.headers['Content-Type'] = 'text/event-stream'
    return response
app.register_blueprint(sse, url_prefix=SSE_SERVER_PATH, headers={'X-Accel-Buffering': 'no'})

# MySQL配置
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQL_CONFIG["user"]}:{MYSQL_CONFIG["password"]}@{MYSQL_CONFIG["host"]}/{MYSQL_CONFIG["database"]}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_POOL_SIZE'] = 100  # 连接池大小
app.config['SQLALCHEMY_POOL_RECYCLE'] = 3600  # 连接池中连接最长使用时间，单位秒
# 创建数据库引擎
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'], poolclass=QueuePool, pool_size=app.config['SQLALCHEMY_POOL_SIZE'], pool_recycle=app.config['SQLALCHEMY_POOL_RECYCLE'], pool_pre_ping=True)
# 创建数据库连接
db = SQLAlchemy(app)
Session = sessionmaker(bind=engine)
db_session = scoped_session(Session)

@app.before_request
def before_request():
    g.session = db_session()

@app.after_request
def after_request(response):
    g.session.close()
    return response

@app.teardown_request
def shutdown_session(exception=None):
    db_session.close()
    db_session.remove()

@app.teardown_appcontext
def shutdown_session2(exception=None):
    db_session.remove()


@app.route('/api/user/signin', methods = ['POST'])
def signin():
    '''
    用户登录
    '''
    if request.method != 'POST':
        return make_response(jsonify({"errcode":50001,"errmsg":"Please use POST method"}), 500)
    try:
        data: dict = request.get_json()
        username: str = data.get('username')
        password: str = data.get('password')
    except:
        return make_response(jsonify({"errcode":50002,"errmsg":"JSON data required"}), 500)
    if username == None or password == None:
        return make_response(jsonify({"errcode":50003,"errmsg":"Please provide username and password"}), 500)
    try:
        userDB = UserDB(db_session)
        r: dict = userDB.check_password(username, password)
        if r == {}:
            return make_response(jsonify({"errcode":50004,"errmsg":"Username Or Password is incorrect"}), 500)
        ip = request.headers.get('X-Forwarded-For')
        if ip:
            ip = ip.split(',')[0]
        else:
            ip = request.headers.get('X-Real-IP')
        if not ip:
            ip = request.remote_addr
        if ip:
            userDB.write_user_ip(username, ip)
    finally:
        db_session.close()
    r.update(get_openai_apikey())
    r['errcode'] = 0
    r['errmsg'] = 'Success'
    response: Response = make_response(jsonify(r), 200)
    response.headers['Cache-Control'] = 'no-cache'
    return make_response(jsonify(r), 200)

def get_openai_apikey() -> dict:
    '''
    用户登录成功后，返回openai的API key给客户端
    '''
    if os.environ.get('OPENAI_API_KEY') == None:
        return {}
    else:
        return {
            "apiKey": encrypt(os.environ['OPENAI_API_KEY']),
            "baseUrl": OPENAI_PROXY_BASEURL['dev'] if os.environ.get('DEBUG_MODE') != None else OPENAI_PROXY_BASEURL['prod']
            }
    
def encrypt(text):
    key = '0123456789abcdef' # 密钥，必须为16、24或32字节
    cipher = AES.new(key.encode(), AES.MODE_ECB)
    text = text.encode('utf-8')
    # 补齐16字节的整数倍
    text += b" " * (16 - len(text) % 16)
    ciphertext = cipher.encrypt(text)
    # 转为 base64 编码
    return base64.b64encode(ciphertext).decode()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
