from plugins.baseplugin import BasePlugin
import requests
import openai
from config import read_config
import time
import pyautogui
import pyperclip
import os
import pdb


@BasePlugin.register_plugin
class AutoGPT(BasePlugin):
    def __init__(self):
        super().__init__(plugin_name="AutoGPT", author="Libaration", version="0.0.1")

    config = read_config()
    openai.api_key = config.get("openai", "api_key")
    messages = []

    prompt2 = "debug prompt"
    prompt = config.get("openai", "prompt")
    messages.append({"role": "system", "content": prompt})
    stopCount = 15

    def addUserMessage(self, message):
        formatted = {"role": "user", "content": message}
        self.messages.append(formatted)

    def addGPTMessage(self, message):
        formatted = {"role": "assistant", "content": message}
        self.messages.append(formatted)

    def sendMessage(self, message):
        pyperclip.copy(message)
        time.sleep(2)
        pyautogui.hotkey("command", "v", interval=0.25)
        pyautogui.press("enter")
        self.stopCount -= 1

    def generateResponse(self, message):
        # print(self.messages)
        # print(message)
        # response = None
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            max_tokens=50,
            temperature=1,
            presence_penalty=2,
            frequency_penalty=2,
        )
        text_response = response.choices[0].message.content
        self.addGPTMessage(text_response)
        self.sendMessage(text_response)
        print(self.messages)

    def onMessageReceive(self, message):
        print("Remaining messages: {}".format(self.stopCount - 1))
        if self.stopCount == 0:
            exit()
        if (
            message.text == ""
            or message.text == None
            or message.text == " "
            or message.chat.rowid
            != 1  # need to make a smarter way to respond to the same chat
        ):
            return
        else:
            if message.is_from_me == 0:
                self.addUserMessage(message.text)

            self.generateResponse(message.text)
            time.sleep(5)
