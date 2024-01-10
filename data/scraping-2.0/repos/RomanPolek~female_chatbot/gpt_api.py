#open openai_key.pk
import openai
from typing import List
from enum import Enum

class Message:
    class Role(Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    def __init__(self, role: Role, content: str, name: str=None) -> None:
        self.role = role
        self.content = content
        self.name = name
        if self.name is not None:
            self.name = self.name.replace(" ", "_").replace(":", "").replace("/", "").replace("(", "").replace(")", "").replace("?", "").replace(",", "").replace(".", "").replace("<", "").replace("!", "").replace("+", "").replace("&", "")

    def to_dict(self):
        result = {
            "role": self.role.value,
            "content": self.content,
            "name": self.name
        }

        if self.name is None:
            del result["name"]

        return result

def get_key():
    with open("openai_key.pk", "r") as f:
        return f.read()

openai.api_key = get_key()

def gpt_respond(messages: List[Message], **kwargs):
    message_dicts = [message.to_dict() for message in messages]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=message_dicts,
        temperature=kwargs.get("temperature", 1),
        n=1,
        stream=False,
        max_tokens=kwargs.get("max_tokens", 200),
    )
    return response["choices"][0]["message"]["content"]