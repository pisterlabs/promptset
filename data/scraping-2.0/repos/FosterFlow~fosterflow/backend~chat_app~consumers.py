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
        self.user_channel = 'None'

        if self.scope["user"] == AnonymousUser():
            self.close()
            return False
        else:
            self.user_channel = str(self.scope["user"].id)

        self.agents = Agent.objects.filter(user_id=self.scope["user"])
        self.available_chats = list(
            # Chat.objects.filter(Q(owner_id__in=self.agents) | Q(addressee_id__in=self.agents)) # To feature
            Chat.objects.filter(owner_id__in=self.agents)
            .values_list('id', flat=True)
        )

        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.user_channel, self.channel_name
        )

        self.accept()

    def disconnect(self, close_code):
        # Leave room group
        async_to_sync(self.channel_layer.group_discard)(
            self.user_channel, self.channel_name
        )

    def chat_message(self, event):
        self.send(text_data=json.dumps(event["text"]))
