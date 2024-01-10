from flask_restful import Resource
from flask import request, jsonify, send_file
import openai
from dotenv import load_dotenv
import os, requests, shutil, datetime

from models.chat import ChatModel

load_dotenv()

CHATGPT_ENV=os.getenv('CHATGPT')
openai.api_key = CHATGPT_ENV

class DalleClothes(Resource):
    def get(self):
        return jsonify({"message": "Response DALLE IMAGE post a text"})
    
    def post(self):
        clothestext = request.get_json()['text']
        username = request.get_json()['username']
        
        print(f'Wanted Clothes Text : {clothestext}')
        response = openai.Image.create(
            prompt=clothestext,
            n=1,
            size="256x256"
        )
        image_url = response['data'][0]['url']
        file_name = f'images/dll_img_{hash(image_url)}.png'
        
        data_chat = {
            "text": request.get_json()['text'],
            "username": username,
            "publish_date": datetime.datetime.now()
        }
        
        chat = ChatModel(**data_chat)  # since parser only takes in username and password, only those two will be added.
        chat.save_to_database()
        
        res = requests.get(image_url, stream = True)
        
        if res.status_code == 200:
            with open(file_name,'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print('Image sucessfully Downloaded: ', file_name)
            return send_file(file_name, mimetype='image/png')
        else:
            print('Image Couldn\'t be retrieved')
            return jsonify({"error": "Image Couldn\'t be retrieved"})