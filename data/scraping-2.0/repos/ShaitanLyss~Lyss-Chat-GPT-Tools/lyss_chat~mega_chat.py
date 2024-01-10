from ast import mod
from json import JSONEncoder
import json
from pprint import pp
from typing import Any, List, Literal
from openai import OpenAI
from datetime import datetime

import sys
from dotenv import load_dotenv
import os
import argparse
from openai.types.chat import ChatCompletionMessageParam

from pydantic import BaseModel, Field

base_dir = os.path.dirname(os.path.realpath(__file__))


class Message(BaseModel):
    role: Literal["user", "system", "assistant", "tool"] | None
    content: str
    created_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context: Any) -> None:
        self.content = self.content.strip()
        return super().model_post_init(__context)


class ChatHistory(BaseModel):
    messages: List[Message] = []

    def add_message(self, message: Message):
        self.messages.append(message)

    @staticmethod
    def from_json(json):
        return ChatHistory(**json)

    def to_json(self):
        return self.model_dump_json(indent=4)

    @staticmethod
    def from_file(filename):
        try:
            with open(filename, "r") as f:
                data = f.read()
                if not data:
                    return ChatHistory()
                return ChatHistory.model_validate_json(data)
        except FileNotFoundError:
            return ChatHistory()

    def to_file(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, "w") as f:
            f.write(self.to_json())

    def __str__(self):
        return "\n".join(
            [f"{m.created_at} {m.role}: {m.content}" for m in self.messages]
        )


custom_instructions: List[Message] = [
    Message(
        role="system",
        content="""
                    My name is Lyss, I am diagnosed with ADHD, suffer of severe fatigue linked to it and bad sleep, so take it into 
                    account when relevant to provide advices or adapt your answers.I am a software engineer,
                    I always yearn for the best and most scalable solution for my designs.
                    I'm also an artist. My personality is INFP. For answering me, use exclusively simple README formatting, no latex. Example en math:
                    écris log(x^3) + 5 pour le logarithme de x au cube plus 5.
""",
    ),
    Message(
        role="system",
        content="""
                    Keep messages short and straight to the point. Make a summary at the end of every message of all the key information from all the conversation, so include the information from the previous summary, enclose the full summary with BEGIN_SUMM and END_SUMM.
                    The goal of this summary is to keep track of the key information of all the messages for the context, it is not displayed to the user.
                """,
    ),
]


def make_openai_message(message: Message) -> ChatCompletionMessageParam:
    return {"content": message.content, "role": message.role}


def chat_with_gpt(message, with_summary=False):
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    chat_history = ChatHistory.from_file(
        os.path.join(base_dir, "save/chat_history.json")
    )

    new_message = Message(role="user", content=message, created_at=datetime.now())
    chat_history.add_message(new_message)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            make_openai_message(msg)
            for msg in custom_instructions + chat_history.messages[-5:]
        ],
        stream=True,
    )
    content_arr = []
    triggered = False
    is_summary = False
    first_chunk = None
    for chunk in response:
        if not first_chunk:
            first_chunk = chunk
        content = chunk.choices[0].delta.content
        # print(triggered, is_summary, content, "|||")
        if content:
            content_arr.append(content)
            if content == "BEGIN":
                triggered = True
            elif triggered and content == "_SUM":
                is_summary = True
            elif not is_summary or with_summary:
                sys.stdout.write(content)
                sys.stdout.flush()
            else:
                # sys.stdout.write("_")
                sys.stdout.flush()
    print()
    if first_chunk:
        new_message = Message(
            role=first_chunk.choices[0].delta.role, content="".join(content_arr), created_at=datetime.now()
        )
        chat_history.add_message(new_message)
        chat_history.to_file(os.path.join(base_dir, "save/chat_history.json"))


def main():
    parser = argparse.ArgumentParser(description="Chat with GPT")
    parser.add_argument(
        "message", type=str, help="The message to chat with GPT", nargs="+"
    )

    parser.add_argument(
        "-s", help="With summary", action="store_true", default=False, dest="summary"
    )
    args = parser.parse_args()
    chat_with_gpt(" ".join(args.message), with_summary=args.summary)


if __name__ == "__main__":
    main()
