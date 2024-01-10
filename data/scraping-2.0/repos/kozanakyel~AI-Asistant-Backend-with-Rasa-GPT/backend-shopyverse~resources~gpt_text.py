from flask_restful import Resource
from flask import request, jsonify
import openai
from dotenv import load_dotenv
import os, datetime

from models.chat import ChatModel

load_dotenv()

CHATGPT_ENV=os.getenv('CHATGPT')
openai.api_key = CHATGPT_ENV

class GptText(Resource):
    def get(self):
        return jsonify({"message": "Response CHATGPT post a text"})
    
    def post(self):
        question = request.get_json()['text']
        username = request.get_json()['username']
        
        print(f'Question : {question}')
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=question,
            max_tokens=256,
            n=1
        )
        
        data_chat = {
            "text": request.get_json()['text'],
            "username": username,
            "publish_date": datetime.datetime.now()
        }
        
        chat = ChatModel(**data_chat)  # since parser only takes in username and password, only those two will be added.
        chat.save_to_database()
        
        response_text = response['choices'][0]['text']
        print("*"*40)
        print(f'GPT TEXT response: {response_text}')
        print("*"*40)
        
        return jsonify(response_text)