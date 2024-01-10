import json
from urllib.parse import parse_qs
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth.models import AnonymousUser
from .serializers import CustomSerializer
from chat.models import Message
from channels.exceptions import DenyConnection
import openai
import asyncio
from decouple import config
from profiles.models import *

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.get_token(self.scope)
        user = self.scope['user']
        if isinstance(user, AnonymousUser):
            raise DenyConnection("Unauthorized")
        # Get current user id to dynamically manage security of creating private rooms
        self.current_user_id = user.id # Store current user id for reference
        project_id = self.scope['url_route']['kwargs']['project_id'] # Get the project id to create a room for this project
        self.client = await self.get_client(self.current_user_id) # Store the client for reference
        self.role = await self.get_user_role(user)
        self.project_status = await self.get_project(self.role,project_id,self.client,user) # Get the project based on the client and project id given

        if self.project_status is None:
            # Check validity of creating rooms based on project of the client
            raise DenyConnection("Project does not exist")
        self.pm_client = await self.get_client(self.project_status)
        self.room_name = ( #Create the room for the specific project
            f'{self.current_user_id}_{self.project_status}_{project_id}'
            if int(self.current_user_id) > int(self.project_status)
            else f'{self.project_status}_{self.current_user_id}_{project_id}'
        )
        self.room_group_name = f'chat_{self.room_name}'
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        await self.send_previous_messages()

    async def disconnect(self, close_code):
        try:
            if hasattr(self, 'room_group_name'):
                await self.channel_layer.group_discard(self.room_group_name, self.channel_layer)
            await super().disconnect(close_code)
        except:
            pass


    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        message = data['message']
        toggle = data['toggle']
        await self.save_message(sender=self.client, message=message, thread_name=self.room_group_name)
        if self.role == 'Client' and toggle == 'ON':
            # Send the user's message immediately
            messages = await self.get_messages()
            await self.send_message_to_group(message, self.current_user_id, messages)
            # Schedule the sending of the response with a delay
            asyncio.create_task(self.send_response_with_delay(message))
        else:
            messages = await self.get_messages()
            await self.send_message_to_group(message, self.current_user_id, messages)

    async def send_response_with_delay(self, message):
        # Get the response from ChatGPTMessage model
        response_message = await self.get_chat_gpt_response(message)
        await self.save_message(sender=self.pm_client, message=response_message, thread_name=self.room_group_name)
        messages = await self.get_messages()
        await self.send_message_to_group(response_message, self.project_status, messages)

    async def send_message_to_group(self, message, user_id, messages):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'user_id': user_id,
                'messages': messages,
            },
        )

    async def create_clean_messages(self, message_list):
        clean_messages = []
        for msg in message_list:
            clean_msg = {
                'message': msg['fields']['message'],
                'thread_name': msg['fields']['thread_name'],
                'timestamp': msg['fields']['timestamp'],
                'client_id': msg['fields']['sender__pk'],
                'sender_email': msg['fields']['sender__email']
            }
            clean_messages.append(clean_msg)
        return clean_messages

    async def chat_message(self, event):
        user_id = event['user_id']
        messages = event['messages']
        # Parse the JSON string into a Python list of dictionaries
        message_list = json.loads(messages)
        clean_messages = await self.create_clean_messages(message_list)

        await self.send(
            text_data=json.dumps(
                {
                    'user_id': user_id,
                    'messages': clean_messages,
                }
            )
        )

    async def send_previous_messages(self):
        messages = await self.get_messages()

        # Parse the JSON string into a Python list of dictionaries
        message_list = json.loads(messages)
        clean_messages = await self.create_clean_messages(message_list)

        await self.send(
            text_data=json.dumps(
                {
                    'user_id': self.current_user_id,
                    'messages': clean_messages,
                }
            )
        )



    async def get_token(self,scope):
        query_string = scope["query_string"]
        query_params = query_string.decode()
        query_dict = parse_qs(query_params)
        token = query_dict["token"][0]
        user = await self.returnUser(token)
        scope["user"] = user

    @database_sync_to_async
    def returnUser(self, token_string):
        try:
            user = User.objects.get(auth_token=token_string)
        except:
            user = AnonymousUser()
        return user

    async def get_project(self,role,project_id,client,user):
        try:
            project = await self.check_project_existance(role,project_id,client,user)
            if role != 'PM':
                return project.id
            user = await self.get_user(project)
            return user.id
        except:
            return None


    @database_sync_to_async
    def check_project_existance(self,role, project_id,client,user):
        try: # If the project exists with the client being the owner then return the project
            if role != 'PM':
                return Project.objects.get(id=project_id, client=client).production_manager
            return Project.objects.get(id=project_id, production_manager=user).client
        except Exception as e: # Else return None to close the connection
            return None

    @database_sync_to_async
    def get_user_role(self,user):
        return Role.objects.get(user=user).role

    @database_sync_to_async
    def get_user(self,client):
        return Client.objects.get(id = client.id).user


    @database_sync_to_async
    def get_client(self, user_id):
        client = User.objects.get(id=user_id)
        return Client.objects.get(user=client)

    @database_sync_to_async
    def get_messages(self):
        custom_serializers = CustomSerializer()
        messages = custom_serializers.serialize(
            Message.objects.select_related().filter(thread_name=self.room_group_name),
            fields=(
                'sender__pk',
                'sender__username',
                'sender__last_name',
                'sender__first_name',
                'sender__email',
                'sender__last_login',
                'sender__is_staff',
                'sender__is_active',
                'sender__date_joined',
                'sender__is_superuser',
                'message',
                'thread_name',
                'timestamp',
            ),
        )
        return messages

    @database_sync_to_async
    def save_message(self, sender, message, thread_name):
        Message.objects.create(sender=sender, message=message, thread_name=thread_name)

    @database_sync_to_async
    def get_chat_gpt_response(self, message):
        try:
            openai.api_key = config("OPENAI_API_KEY")
            completion = openai.Completion.create(
                prompt=f"{ChatGPTMessage.objects.last().message}\n\nQ:{message}?\nA:",
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.7,
                model="text-davinci-003",
            )

            print(f"passed 3\n")
            ans = completion.choices[0].text.strip()
            print(ans)
            if not ans or ans == "":
                ans = "I don't understand. What did you say? Try with another message."
            return ans
        except:
            pass
