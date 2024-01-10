import openai
from channels.generic.websocket import AsyncWebsocketConsumer
import json


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = 'chat'
        self.room_group_name = 'chat_group'

        # 加载openai api key
        openai.api_key = "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"

        # 加入聊天室
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data=None, bytes_data=None):
        message = json.loads(text_data)['message']
        print("Received message:", message)

        # 调用ChatGPT API
        response = openai.Completion.create(
            engine="davinci",
            prompt=message,
            max_tokens=60
        )

        # 发送ChatGPT生成的回复到聊天室
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': response.choices[0].text
            }
        )

    async def chat_message(self, event):
        message = event['message']

        # 发送消息到WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))
