from flask import Flask, jsonify, request
from flask_cors import CORS
import openai
import re
import os

app = Flask(__name__)

CORS(app)

API_KEY = "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" # Put new API Key
openai.api_key = API_KEY


@app.route('/ask', methods=['POST'])
def ask_openai():
    user_input = "Hello, How are you?"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    text = response.choices[0].message['content']
    
    return jsonify(respond=text)


if __name__ == '__main__':
    app.run(debug=True, port=5000)