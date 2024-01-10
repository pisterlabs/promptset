import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer

class OpenAIConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        if message == 'start':
            # Simulate streaming from OpenAI (you'd replace with the OpenAI API call)
            for i in range(10):
                await self.send(text_data=json.dumps({
                    'message': f'Number {i}'
                }))
                await asyncio.sleep(0.5)