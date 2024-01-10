from flask import Flask,render_template, Blueprint, url_for, request,jsonify,redirect
from blueprints.database.mongoapi import database
from blueprints.database.mongodb import writehistory
import requests
import openai
import os

gptapi = Blueprint('gptapi',__name__)
gptapi.register_blueprint(database)

openai.api_key = os.getenv('GPTKEY')

#site/gpt/query/?prompt={prompt}&email={email}
@gptapi.route('/query/', methods=['POST'])
def madeQuery():
    prompt=request.form.get('query')

    query = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.8,
        max_tokens=100
    )

    data = {
        'email': request.args.get('email'),
        'query': prompt,
        'response': query['choices'][0]['text']
    }

    writehistory(data)

    return redirect('/')