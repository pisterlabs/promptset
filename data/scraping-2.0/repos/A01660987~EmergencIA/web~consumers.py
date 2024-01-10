from channels.generic.websocket import AsyncWebsocketConsumer
from dotenv import load_dotenv
from deepgram import Deepgram
from typing import Dict
import json
import openai
from decouple import config
from asgiref.sync import sync_to_async
import asyncio
import os

load_dotenv()
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

class gptConsumer(AsyncWebsocketConsumer):

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
        

class TranscriptConsumer(AsyncWebsocketConsumer):
   dg_client = Deepgram(os.getenv('DEEPGRAM_API_KEY'))


   async def receive(self, bytes_data):
     self.socket.send(bytes_data)

   async def get_transcript(self, data: Dict) -> None:
       if 'channel' in data:
           transcript = data['channel']['alternatives'][0]['transcript']

           if transcript:
               await self.send(transcript)


   async def connect_to_deepgram(self):
       try:
           self.socket = await self.dg_client.transcription.live({
               'punctuate': True,
               'interim_results': False,
               'smart_format': True,
               'language': 'es',
               'model': 'nova',
          })
           self.socket.registerHandler(self.socket.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
           self.socket.registerHandler(self.socket.event.TRANSCRIPT_RECEIVED, self.get_transcript)

       except Exception as e:
           raise Exception(f'Could not open socket: {e}')

   async def connect(self):
       await self.connect_to_deepgram()
       await self.accept()


   async def disconnect(self, close_code):
       await self.channel_layer.group_discard(
           self.room_group_name,
           self.channel_name
       )
