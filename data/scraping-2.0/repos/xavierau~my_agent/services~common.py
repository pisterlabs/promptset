import json
import os
import json
import uuid
from datetime import datetime
import enum
from typing import Optional, Any, List
import pytz

from openai import OpenAI
from playsound import playsound
from pydantic import BaseModel, Field

import dotenv
from sqlalchemy.orm import Session
from database.schemas import Message as DBMessage, MessageCreate

import config
from database.utils import get_messages_by_session_id, create_message
from tools.common import Tool
from tools.email_writer import EmailWriter
from tools.google_search import GoogleSearchTool
from tools.note import WriteToMyNoteTool, ReadFromMyNoteTool

dotenv.load_dotenv()


class Message(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: uuid.uuid4().hex)
    role: str
    content: Optional[str] | Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    function: Optional[Any] = None
    name: Optional[str] = None
    function_response: Optional[Any] = None
    frontend_render: Optional[Any] = None

    def to_json(self):
        obj_dict = Message(role="user").__dict__
        keys = list(obj_dict.keys())
        temp = {}
        for key in keys:
            if key == "created_at":
                temp[key] = self.created_at
            elif self.__dict__.get(key) is not None:
                temp[key] = self.__dict__.get(key)

        return temp


class History(BaseModel):

    def get_messages(self) -> List[Message]:
        raise NotImplementedError

    def get_latest_messages(self) -> Message:
        raise NotImplementedError

    def add_message(self, message: Message):
        raise NotImplementedError

    def to_openai_list(self, limit=20) -> List[Any]:
        temp_list: List[dict] = []
        obj_dict = Message(role="user").__dict__
        keys = list(obj_dict.keys())
        messages = self.get_messages()
        messages = messages[-limit:]

        exclude_keys = [
            "created_at",
            "id"
        ]

        for m in messages:
            temp = {}
            for key in keys:
                val = m.__dict__.get(key)
                if key not in exclude_keys and val is not None and val != "":
                    temp[key] = m.__dict__.get(key)
            temp_list.append(temp)

        sanitized_temp_list = self._sanitize_list(temp_list)

        return sanitized_temp_list

    def _sanitize_list(self, temp_list):
        first_item = temp_list[0]
        if first_item.get("role") == "tool":
            temp_list = temp_list[1:]
            return self._sanitize_list(temp_list)
        else:
            return temp_list


class PostgresHistory(History):
    session: Session
    session_id: uuid.UUID

    class Config:
        arbitrary_types_allowed = True

    def get_latest_messages(self) -> Message:
        messages: List[DBMessage] = get_messages_by_session_id(db=self.session,
                                                               session_id=self.session_id,
                                                               limit=1)
        return Message(**messages[0].message)

    def get_messages(self, **kwargs) -> List[Message]:
        messages: List[DBMessage] = get_messages_by_session_id(db=self.session,
                                                               session_id=self.session_id,
                                                               skip=kwargs.get("skip", 0),
                                                               limit=kwargs.get("limit", 20))

        messages.reverse()

        return [Message(**(m.message)) for m in messages]

    def add_message(self, message: Message):
        data = {
            "message": dict(message),
            "role": message.role,
            "session_id": self.session_id,
        }

        message_create = MessageCreate(**data)

        create_message(db=self.session, message=message_create)


class FileBaseHistory(History):
    file_path: str

    def get_latest_messages(self) -> Message:
        return self.get_messages()[-1]

    def get_messages(self) -> List[Message]:
        with open(self.file_path, 'r') as fp:
            content = fp.read()
            try:
                json_data = json.loads(content)
                return [Message(**d) for d in json_data]
            except:
                return []

    def add_message(self, message: Message):
        messages = self.get_messages()
        messages.append(message)

        with open(self.file_path, 'w') as fp:
            json.dump([m.to_json() for m in messages], fp)


class SimpleHistory(History):
    messages: List[Message] = []

    def get_latest_messages(self) -> Message:
        return self.messages[-1]

    def get_messages(self) -> List[Message]:
        return self.messages

    def add_message(self, message: Message):
        self.messages.append(message)
        return self


class Memory(BaseModel):
    history: History = Field(default_factory=SimpleHistory)

    def get_latest_message(self) -> Message:
        return self.history.get_latest_messages()

    def add(self, message: Message):
        self.history.add_message(message)
        return self

    def to_list(self):
        return self.history.to_openai_list()


class SystemPromptBuilder:
    memory: Optional[Memory] = None
    _agent_setting: Optional[str] = None
    _user_setting: Optional[str] = None
    _other_setting: Optional[str] = None

    def set_memory_setting(self, memory: Memory):
        self._memory = memory
        return self

    def set_agent_setting(self, agent_setting: str):
        self._agent_setting = agent_setting
        return self

    def set_user_setting(self, user_setting: str):
        self._user_setting = user_setting
        return self

    def set_other_setting(self, other_setting: str):
        self._other_setting = other_setting
        return self

    def build(self) -> str:
        return "\n\n".join(list(filter(lambda x: x, [
            self._agent_setting,
            self._user_setting,
            self._other_setting
        ])))
