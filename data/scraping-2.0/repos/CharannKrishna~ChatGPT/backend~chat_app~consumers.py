import json
import os
import openai
import environ

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from project.settings import BASE_DIR
from django.db.models import Q
from chat_app.models import Chat, Message
from user_app.models import Agent

env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))
openai.api_key = env('OPENAI_API_KEY')


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["chat_id"]
        self.chat_id = int(self.room_name)
        self.chat = Chat.objects.get(id=self.chat_id)

        if self.scope["user"] == AnonymousUser():
            self.close()
            return False

        self.agents = Agent.objects.filter(user_id=self.scope["user"])
        available_chats = list(
            # Chat.objects.filter(Q(owner_id__in=self.agents) | Q(addressee_id__in=self.agents)) # To feature
            Chat.objects.filter(owner_id__in=self.agents)
            .values_list('id', flat=True)
        )

        if self.chat_id not in available_chats:
            self.close()
            return False

        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.room_name, self.channel_name
        )
        self.owner_chat = Agent.objects.get(id=self.chat.owner_id.id)
        self.addressee = Agent.objects.get(id=self.chat.addressee_id.id)

        self.accept()

    def disconnect(self, close_code):
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.room_name, self.channel_name
        )

    # Receive message from WebSocket
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        try:
            prompt = text_data_json['prompt']
            chat_id = Chat.objects.get(id=text_data_json['chat_id'])
            owner_id = Agent.objects.get(id=text_data_json['owner_id'])
            method = text_data_json['method']
            if prompt and method == 'request' and chat_id == self.chat and Agent.objects.filter(id=owner_id.id).exists():
                message1 = Message.objects.create(chat_id=chat_id,
                                                  message_text=prompt,
                                                  owner_id=owner_id)
                self.send(text_data=json.dumps(
                    {
                        "id": message1.id,
                        "message_text": message1.message_text,
                        "created_at": str(message1.created_at),
                        "chat_id": message1.chat_id.id,
                        "owner_id": message1.owner_id.id,
                        "method": "response"
                    }
                ))

                message2 = Message.objects.create(chat_id=self.chat,
                                                  message_text='',
                                                  owner_id=self.addressee)
                complete = self.ask_gpt_stream(prompt, message2)
                message2.message_text = complete
                message2.save()
            elif method != 'request' or method != 'response':
                self.send(text_data=json.dumps(
                    {
                        'error': 'Not request or response'
                    }))
        except Exception as e:
            self.send(text_data=json.dumps(
                {
                    'error': str(e)
                }))

    # Receive message from room group
    def ask_gpt_stream(self, prompt, main_message):
        previous_messages = Message.objects.filter(chat_id=self.chat_id).order_by('-id')[:5]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        for message in previous_messages:
            if message.owner_id == self.owner_chat:
                mes = {"role": "user", "content": message.message_text}
            else:
                mes = {"role": "assistant", "content": message.message_text}
            messages.append(mes)

        user_prompt = {"role": "user", "content": prompt}
        messages.append(user_prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        complete = ""
        self.send(text_data=json.dumps({
            'chat_id': main_message.chat_id.id,
            'created_at': str(main_message.created_at),
            'id': main_message.id,
            'message_chunk': "",
            'owner_id': main_message.owner_id.id,
            'status': "start"
        }))
        # iterate through the stream of events
        for event in response:

            try:
                if event['choices'][0]['finish_reason'] != 'stop':
                    event_text = event['choices'][0]['delta']['content']  # extract the text
                    self.send(text_data=json.dumps(
                        {
                            'id': main_message.id,
                            "message_chunk": event_text,
                            "status": "progress"
                        }))
                    complete += event_text
                else:
                    self.send(text_data=json.dumps(
                        {
                            'id': main_message.id,
                            "message_chunk": '',
                            "status": "done"
                        }))
            except Exception as e:
                self.send(text_data=json.dumps(
                    {
                        'id': main_message.id,
                        "message_chunk": '',
                        "status": "error"
                    }))

        return complete
