from channels.generic.websocket import AsyncWebsocketConsumer
import json
from .models import Chat
import openai
from decouple import config
from asgiref.sync import sync_to_async
import asyncio

openai.api_key = config("OPENAI_API_KEY")

async def generate_answer(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response["choices"][0]["text"]

class ChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        question = text_data_json['question']
        chat = Chat(question=question)
        await sync_to_async(chat.save)()
        answer = await generate_answer(question)
        chat.answer = answer
        await sync_to_async(chat.save)()
        response = {
            'question': question,
            'answer': answer
        }
        await self.send(text_data=json.dumps(response))