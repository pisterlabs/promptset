import base64
import json
import os

import openai
from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from managements.views import cropping_image
from .models import Room, Message
from .templatetags.time_filters import get_time_since


class ChatExpert(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.room_id = None
        self.room_group_name = None

    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f"chat_room_{self.room_id}"

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
        if text_data:
            data = json.loads(text_data)
            room = await self.get_room(self)
            user = await self.get_room_user(self, room)
            expert = await self.get_room_expert(self, room)
            response = {"type": "chat_message"}
            account = None
            if user.id == int(data["user_id"]):
                account = user
                response["from"] = "user"
            elif expert.id == int(data["user_id"]):
                account = expert
                response["from"] = "expert"

            if account:
                img_flag = False
                img_dir_filepath = ''
                temp_img_filename = ''
                message = None
                if data["imgBase64Str"] != '' and data["imgFormat"] != '':
                    img_dir_filepath = os.path.join(settings.MEDIA_ROOT, "message_img")
                    os.makedirs(img_dir_filepath, exist_ok=True)
                    temp_img_filename = "temp_message_img" + '.' + data["imgFormat"]
                    img_path = os.path.join(img_dir_filepath, temp_img_filename)
                    decoded_data = base64.b64decode(data["imgBase64Str"])
                    with open(img_path, 'wb') as f:
                        f.write(decoded_data)

                    img_flag = True

                    if data["content"] != '':
                        message = await self.create_message(self, room, account, "message_img" + "/" + temp_img_filename, data["content"])
                    else:
                        message = await self.create_message(self, room, account, "message_img" + "/" + temp_img_filename)
                elif data["content"] != '':
                    message = await self.create_message(self, room, account, None, data["content"])

                if img_flag and message:
                    img_filename = str(message.id) + "_message_img.jpg"
                    cropping_image(data["cropping_details"], img_dir_filepath, img_filename, temp_img_filename)
                    message.image = "message_img" + "/" + img_filename
                    await self.save_message(self, message)

                html_format = ''
                if message:
                    html_content = ''
                    html_img = ''
                    if img_flag:
                        html_img = "<img class='w-25 py-2' src=" + str(message.image.url) + ">"
                    if data["content"]:
                        html_content = "<h5 class='py-2'>" + data["content"] + "</h5>"

                    if response["from"] == "user":
                        html_format = "<div class='chat-user-style mb-3'><div class='text-center'>" + html_img + html_content + "</div><div class='d-flex px-3 py-1'><div class='d-md-flex me-auto align-items-md-center'><span>send " + get_time_since(
                            message.timestamp) + "</span></div><div class='text-end d-md-flex ms-auto align-items-md-center'><a class='d-md-flex justify-content-md-end align-items-md-center' href='#'><img class='questions-profile-img me-2' src=" + account.pimg.url + ">" + account.fname + " " + account.lname + "</a></div></div></div>"
                    else:
                        html_format = "<div class='chat-expert-style mb-3'><div class='text-center'>" + html_img + html_content + "</div><div class='d-flex px-3 py-1'><div class='text-start d-md-flex me-auto align-items-md-center'><a href='#'><img class='questions-profile-img me-2' src=" + account.pimg.url + ">" + account.fname + " " + account.lname + "</a></div><div class='d-md-flex ms-auto align-items-md-center'><span>received " + get_time_since(
                            message.timestamp) + "</span></div></div></div>"

                response["html_format"] = html_format

                await self.channel_layer.group_send(
                    self.room_group_name,
                    response
                )

    async def chat_message(self, event):
        await self.send(text_data=json.dumps(event))

    @sync_to_async
    def get_room(self, arg):
        return Room.objects.get(id=self.room_id)

    @sync_to_async
    def get_room_user(self, arg, room):
        return room.user

    @sync_to_async
    def get_room_expert(self, arg, room):
        return room.expert

    @sync_to_async
    def create_message(self, arg, room, account, img_path=None, content=None):
        message = None
        if img_path:
            if content:
                message = Message.objects.create(room=room, account=account, image=img_path, content=content)
            else:
                message = Message.objects.create(room=room, account=account, image=img_path)
        elif content:
            message = Message.objects.create(room=room, account=account, content=content)

        return message

    @sync_to_async
    def save_message(self, arg, message):
        return message.save()


class ChatAI(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.room_id = None
        self.room_group_name = None

    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f"chat_room_{self.room_id}"

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
        if text_data:
            data = json.loads(text_data)
            room = await self.get_room(self)
            user = await self.get_room_user(self, room)
            response = {"type": "chat_message"}
            account = None
            if user.id == int(data["user_id"]):
                account = user
                response["from"] = "user"

            if account:
                img_flag = False
                message = None
                if data["content"] != '':
                    message = await self.create_message(self, room, account, data["content"])

                    html_format = ''
                    if message and response["from"] == "user":
                        html_format = "<div class='chat-user-style mb-3'><div class='text-center'><h5 class='py-2'>" + data[
                            "content"] + "</h5></div><div class='d-flex px-3 py-1'><div class='d-md-flex me-auto align-items-md-center'><span>send " + get_time_since(
                            message.timestamp) + "</span></div><div class='text-end d-md-flex ms-auto align-items-md-center'><a class='d-md-flex justify-content-md-end align-items-md-center' href='#'><img class='questions-profile-img me-2' src=" + account.pimg.url + ">" + account.fname + " " + account.lname + "</a></div></div></div>"

                    response["html_format"] = html_format

                    await self.channel_layer.group_send(
                        self.room_group_name,
                        response
                    )

                    openai.api_key = os.environ['OPENAI_API_KEY']

                    ai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "user",
                                "content": data["content"] + " in summary"
                            }
                        ],
                        temperature=1,
                        max_tokens=1275,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )

                    ai_content = ai_response.choices[0].message.content
                    ai_content = ai_content.replace("\\n", "<br>")
                    ai_content = ai_content.replace("'''", "<br><br>")
                    ai_content = ai_content.strip()

                    ai_img_path = f"{settings.STATIC_URL}assets/img/ai.jpg"
                    html_format = "<div class='chat-expert-style mb-3'><div class='text-center'><h5 class='py-2'>" + ai_content + "</h5></div><div class='d-flex px-3 py-1'><div class='text-start d-md-flex me-auto align-items-md-center'><a href='#'><img class='questions-profile-img w-30-px me-2' src=" + ai_img_path + ">AI (ChatGPT)</a></div><div class='d-md-flex ms-auto align-items-md-center'><span>received " + get_time_since(
                        message.timestamp) + "</span></div></div></div>"

                    response["from"] = "ai"
                    response["html_format"] = html_format

                    await self.create_message(self, room, None, ai_content)

                    await self.channel_layer.group_send(
                        self.room_group_name,
                        response
                    )

    async def chat_message(self, event):
        await self.send(text_data=json.dumps(event))

    @sync_to_async
    def get_room(self, arg):
        return Room.objects.get(id=self.room_id)

    @sync_to_async
    def get_room_user(self, arg, room):
        return room.user

    @sync_to_async
    def create_message(self, arg, room, account=None, content=None):
        message = None
        if account:
            message = Message.objects.create(room=room, account=account, content=content)
        else:
            message = Message.objects.create(room=room, content=content)

        return message

    @sync_to_async
    def save_message(self, arg, message):
        return message.save()
