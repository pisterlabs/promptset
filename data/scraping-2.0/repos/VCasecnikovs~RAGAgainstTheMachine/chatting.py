from enum import Enum
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


def get_openAI_client():
    load_dotenv()

    client = OpenAI()
    return client


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


def chat_inference(
    messages: list[ChatMessage],
    client: OpenAI,
    model="gpt-4-1106-preview",
):
    formatted_messages = []
    for message in messages:
        formatted_messages.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    completion = client.chat.completions.create(
        response_format={"type": "json_object"},
        model=model,
        messages=[
            *formatted_messages,
        ],
    )

    model_answer = completion.choices[0].message.content
    return model_answer
