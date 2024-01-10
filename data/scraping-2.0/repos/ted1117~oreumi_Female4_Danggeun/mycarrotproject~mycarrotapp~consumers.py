import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import Chat
import openai
from django.conf import settings
import asyncio

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = "chat_%s" % self.room_name
        openai.api_key = settings.OPENAI_KEY

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # @sync_to_async
    def create_chat_message(self, message, is_read):
        Chat.objects.create(content=message, is_read=is_read)

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        if "mark_as_read" in text_data_json:
            await self.mark_as_read(text_data_json) 
        else:
            message = text_data_json["message"]
            username = text_data_json["username"]  # username을 추출
            time = text_data_json["time"]
            isGPT = text_data_json["isGPT"]

            new_msg = await self.create_chat(message, is_read=False)
            
                # Send message to room group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "chat_message",
                    "message": message,
                    "username": username,
                    "time": time,
                    "message_id": new_msg.id
                },
            )

            if isGPT:
                await self.get_GPT_response(message, time)
        
        '''
        await self.create_chat(message, is_read=True)

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "chat_message",
                "message": message,
                "username": username,  # username을 함께 전송
                "time": time,
            },
        )

        if isGPT:
            await self.get_GPT_response(message, time)
        '''

    async def mark_as_read(self, event):
        message_id = event["message_id"]
        message_sender = event["username"]
        message = await self.get_message_by_id(message_id)

        # 수정된 부분: 메시지를 읽었을 때에만 is_read를 True로 설정
        if message:
            message.is_read = True
            await self.save_message(message)

        # 수정된 부분: 읽음 상태를 브로드캐스팅
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "read_status",
                "message_id": message_id,
                "message_sender": message_sender
            }
        )

    # Receive read status from room group
    async def read_status(self, event):
        message_id = event["message_id"]
        message_sender = event["message_sender"]

        # Send read status to WebSocket
        await self.send(
            text_data=json.dumps(
                {
                    "read_status": True,
                    "message_id": message_id,
                    "message_sender": message_sender
                }
            )
        )
        
    async def get_GPT_response(self, user_message, time):
        query = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0.7
        )
        response = query['choices'][0]['message']['content'].strip()

        await self.create_chat(response)

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "chat_message",
                "message": response,
                "username": "ChatGPT",
                "time": time,
            },
        )

    # stackoverflow
    from channels.db import database_sync_to_async

    @database_sync_to_async
    def create_chat(self, message, is_read):
        return Chat.objects.create(content=message, is_read=is_read)

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]
        username = event["username"]  # username을 추출
        time = event["time"]
        message_id = event["message_id"]
        #isGPT = event.get("isGPT", False)
        #new_msg = await self.create_chat(message, is_read=True)

        # Send message and username to WebSocket
        await self.send(
            text_data=json.dumps(
                {
                    "message": message,
                    "username": username,
                    "time": time,
                    "message_id": message_id
                    #"isGPT": isGPT
                }  # username도 함께 전송
            )
        )

    @database_sync_to_async
    def get_message_by_id(self, message_id):
        try:
            return Chat.objects.get(id=message_id)
        except Chat.DoesNotExist:
            return None

    @database_sync_to_async
    def save_message(self, message):
        message.save()