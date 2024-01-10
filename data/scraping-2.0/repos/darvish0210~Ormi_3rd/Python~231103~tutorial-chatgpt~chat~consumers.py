from typing import List
import openai

from channels.generic.websocket import JsonWebsocketConsumer
from django.contrib.auth.models import AbstractUser

from chat.models import RolePlayingRoom, GptMessage

class RolePlayingRoomConsumer(JsonWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpt_messages: List[GptMessage] = []
    

    def connect(self):
        room = self.get_room()
        if room is None:
            self.close()
        else:
            self.accept()

            self.gpt_messages = room.get_initial_messages()

            # 🔥 추가) 기존 self.gpt_messages 기반으로 GPT API 호출 응답을 웹클라이언트에게 전달합니다.
            # openai api 호출 시에 예외가 발생하면 type=openai-error 응답을 주도록 합니다.
            try:
                assistant_message = self.gpt_query()
            except openai.error.OpenAIError as e:
                self.send_json({
                    "type": "openai-error",
                    "message": str(e),
                })
            else:
                self.send_json({
                    "type": "assistant-message",
                    "message": assistant_message,
                })
    def get_room(self) -> RolePlayingRoom | None:
        user: AbstractUser = self.scope["user"]
        room_pk = self.scope["url_route"]["kwargs"]["room_pk"]
        room: RolePlayingRoom = None

        if user.is_authenticated:
            try:
                room = RolePlayingRoom.objects.get(pk=room_pk, user=user)
            except RolePlayingRoom.DoesNotExist:
                pass

        return room
    def gpt_query(self, command_query: str = None, user_query: str = None) -> str:
        if command_query is not None and user_query is not None:
            raise ValueError("command_query 인자와 user_query 인자는 동시에 사용할 수 없습니다.")
        elif command_query is not None:
            self.gpt_messages.append(GptMessage(role="user", content=command_query))
        elif user_query is not None:
            self.gpt_messages.append(GptMessage(role="user", content=user_query))
    
        response_dict = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.gpt_messages,
            temperature=1,  # 디폴트 값: 1
        )
        response_role = response_dict["choices"][0]["message"]["role"]
        response_content = response_dict["choices"][0]["message"]["content"]
    
        # command_query 수행 시에는 응답을 self.gpt_messages에 추가하지 않습니다. 그 외에는 채팅 내역으로서 추가합니다.
        if command_query is None:
            gpt_message = GptMessage(role=response_role, content=response_content)
            self.gpt_messages.append(gpt_message)
    
        # GPT API의 content 응답을 반환합니다. (assistant role?)
        return response_content
    def receive_json(self, content_dict, **kwargs):
        if content_dict["type"] == "user-message":
            assistant_message = self.gpt_query(user_query=content_dict["message"])
            self.send_json({
                "type": "assistant-message",
                "message": assistant_message,
            })
        else:
            self.send_json({
                "type": "error",
                "message": f"Invalid type: {content_dict['type']}",
            })