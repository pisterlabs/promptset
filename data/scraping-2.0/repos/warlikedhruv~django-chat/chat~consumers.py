import json

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
import openai
from django.conf import settings
openai.api_key = settings.OPEN_AI_KEY
model_key = settings.MODEL_KEY
bot_name = settings.BOT_NAME
from . import tasks



class ChatConsumer(WebsocketConsumer):
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        response_message = message
        async_to_sync(self.channel_layer.send)(
            self.channel_name,
            {
                'type': 'chat_message2',
                'message': response_message
            }
        )
        try:
            temp = openai.Completion.create(
                prompt=str(response_message)+"\n\n###\n\n", stop=[" #END", "\n\n###\n\n"], temperature=0.7,
                top_p=1, frequency_penalty=1, presence_penalty=0.6, best_of=1,
                max_tokens=50, model=str(model_key)
            )
            try:
                rsp_msg = temp["choices"][0]["text"].split(":", 1)[1]
            except:
                rsp_msg = temp["choices"][0]["text"].strip()
            response_message = rsp_msg
        except Exception as e:
            response_message = e
        async_to_sync(self.channel_layer.send)(
                self.channel_name,
                {
                    'type': 'chat_message',
                    'message': response_message
                }
            )

    def chat_message2(self, event):
        message = event['message']

        # Send message to WebSocket
        self.send(text_data=json.dumps({
            'message': f'[you]: {message}'
        }))

    def chat_message(self, event):
        message = event['message']

        # Send message to WebSocket
        self.send(text_data=json.dumps({
            'message': f'[{bot_name}]: {message}'
        }))
