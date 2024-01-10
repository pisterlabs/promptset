from flask_restful import Resource
from flask import Response, request, send_file
import json
from configparser import ConfigParser
import jwt
import os
import time
import openai

# opt
conn = ConfigParser()
conn.read("./conf.ini")

# token config
headers = {
  "alg": "HS256",
  "typ": "JWT"
}
salt = conn.get('key','token_key')
gpt_key = conn.get('gpt','api_key')
organization = conn.get('gpt','organization')
openai.api_key = gpt_key
openai.organization = organization

messages = [{"role": "system", "content": "You are a helpful assistant to answer and chat."}]

def validate_user_token(token, username):
    try:
        info = jwt.decode(jwt = token, key = salt, verify=True, algorithms='HS256')
        if username == info["username"]:
            return True
        else:
            return False
    except Exception as e:
        print(repr(e))
        return False

class GDPBot(Resource):
    def get(self):
        pass

    def post(self):
        # form = eval(str(request.data, encoding = "utf-8"))
        form = request.form
        res = None
        if form["request"] == "chat":
            question = form['question']
            messages.append({"role": "user", "content": question})
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = messages)
            reply = response["choices"][0]["message"]["content"]
            messages.append({"role": "system", "content": reply})
            print("\n" + reply + "\n")
            res = Response(response = json.dumps({'status': 0, 'msg': reply}), status = 200, mimetype="application/json")
        elif form["request"] == "reset":
            messages.clear()
            messages.append({"role": "system", "content": "You are a helpful assistant to answer and chat."})
            res = Response(response = json.dumps({'status': 0, 'msg': 'Reset successfully!'}), status = 200, mimetype="application/json")
        '''
        if not validate_user_token(form['token'], username):
            res = Response(response = json.dumps({'status': 2, 'msg': 'Invalid operations! Will come back to the login page.'}), status = 200, mimetype="application/json")
        else:
            exp_time = int(time.time() + 60*60*24*7)
            payload = {"username": username, "path": form['path'], "exp": exp_time}
            token = jwt.encode(payload=payload, key=salt, algorithm='HS256', headers=headers)
            res = Response(response = json.dumps({'status': 0, 'msg': 'Share successfully!', 'token': token}), status = 200, mimetype="application/json")
        '''
        return res