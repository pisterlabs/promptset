from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from typing import Union
import uvicorn
from fastapi import FastAPI
from typing import (
    Dict,
    List,
    Optional
)
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
class Question(BaseModel):
	system_message: Optional[str] = "Ты полезный ассистент, который пишет статьи для блога"
	human_message: Optional[str] = "Помоги написать статью на тему - как правильно изучать английский язык"

class QuestionWithContext(BaseModel):
	system_message: Optional[str] = "Ты полезный ассистент"
	human_message: Optional[str] = "Как правильно изучать английский язык"
	user_id: Optional[str] = "1"
class User(BaseModel):
	user_id: Optional[str] = "1"
class KandinskyModel(BaseModel):
	query: Optional[str] = ""
payloads_dict = {}



app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


credential = "MjUwODlkODMtOTE4NS00MmFhLTlmOTAtNDBjYzE2YTQwZTk4OmEzYTBhYjNjLWM3ZjYtNDViNy1hMWI1LTMzZjQ5OTVlMjcyYw=="

@app.post("/api/getAnswer")
def getAnswer(question:Question):
	with GigaChat(credentials=credential, verify_ssl_certs=False) as giga:
		payload = Chat(
			messages=[
				Messages(
					role=MessagesRole.SYSTEM,
					content=question.system_message
				)
			],
			temperature=0.7,
			# max_tokens=1000,
		)
		payload.messages.append(Messages(role=MessagesRole.USER, content=question.human_message))
		response = giga.chat(payload)
		return {"text": response.choices[0].message}
	print(question.system_message)
	print(question.human_message)
	print(answer)
	return {"text": answer}

@app.post("/api/getAnswerWithContext")
def getAnswerWithContext(question:QuestionWithContext):
	with GigaChat(credentials=credential, verify_ssl_certs=False) as giga:
		payload = getPayload(question.user_id, question.system_message)
		payload.messages.append(Messages(role=MessagesRole.USER, content=question.human_message))
		response = giga.chat(payload)
		payload.messages.append(response.choices[0].message)
		print("Bot: ", response.choices[0].message.content)
		print("payload.messages: ", payload.messages)
		return {"text": response.choices[0].message}

@app.post("/api/clearUserContext")
def clearUserContext(user:User):
	if user.user_id in payloads_dict:
		del payloads_dict[user.user_id]
	return {"status": "ok"}
	
def getPayload(user_id, system_message):
	payload = Chat(
			messages=[
				Messages(
					role=MessagesRole.SYSTEM,
					content=system_message
				)
			],
			temperature=1.2,
			# max_tokens=1000,
		)
	if user_id in payloads_dict:
	    payload = payloads_dict[user_id]
	else:
		payloads_dict[user_id] = payload
	
	return payload








import requests

import json
import time
import base64

from random import randint as r
from random import choice as ch

import os

class Text2ImageAPI:

    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, model, images=1, width=1024, height=576):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            if data['status'] == 'DONE':
                return data['images']

            attempts -= 1
            time.sleep(delay)

def gen(prom, dirr = "res"):
    api = Text2ImageAPI('https://api-key.fusionbrain.ai/', '87B16E11B6552CFD41546754FB79A8AB', '1CF4C80F99ECFDBD5AFEE247BA6DBC71')
    model_id = api.get_model()
    uuid = api.generate(prom, model_id)
    images = api.check_generation(uuid)    

    # Здесь image_base64 - это строка с данными изображения в формате base64
    image_base64 = images[0]

    # Декодируем строку base64 в бинарные данные
    image_data = base64.b64decode(image_base64)

    return image_base64
    # # Открываем файл для записи бинарных данных изображения
    # try:
    #     with open(f"{dirr}/{prom.split('.')[0]} _ {r(0, 100000)}.jpg", "wb") as file:
    #         file.write(image_data)
    # except:
    #     with open(f"{dirr}/{prom.split('.')[0]} _ {r(0, 100000)}.jpg", "w+") as file:
    #         file.write(image_data)


@app.post("/api/genImage")
def genImage(kandinskyModel:KandinskyModel):
	query = kandinskyModel.query
	image = gen(query.replace("\n", " "), query.replace("\n", " ").split(".")[0])
	
	
	return {"status": "ok", "image": image}
	


uvicorn.run(app, port=8080, host='0.0.0.0')
