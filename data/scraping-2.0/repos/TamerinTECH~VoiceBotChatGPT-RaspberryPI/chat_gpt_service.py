import json
from input_listener import InputListener
import struct
import os
import openai
import time


config = json.load(open("config.json"))
openai.api_key = config["openai_key"]
if "openai_org" in config:
    openai.organization = config["openai_org"]


class ChatGPTService:
    def __init__(self, prompt="You are a helpful assistant."):
        self.history = [{"role": "system", "content": prompt}]

    def send_to_chat_gpt(self, message):
        self.history.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.history
        )
        self.history.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        return str.strip(response["choices"][0]["message"]["content"])
