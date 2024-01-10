import os
import time

from mainApp.AI import secretKeys

print("Loading langchain.schema...")
timeStart = time.time()
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



timeToLoadLib = time.time()

print(f"elapse time: {round((timeToLoadLib - timeStart), 2)}")
timeStart = time.time()
print("Loading langchain.chat_models...")

from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = secretKeys.openAi
timeToLoadLib = time.time()
print(f"elapse time: {round((timeToLoadLib - timeStart), 2)}")


class ChatGptLC:
    temperature = 0.7
    promptTemplate = "ChatOpenAI"
    max_tokens = 800
    num_responses = 3

    def __init__(self):
        self.messageHistory = []
        self.chat = ChatOpenAI()

    def __del__(self):
        pass

    def setTemp(self, value):
        self.temperature = value

    def getAnswer(self, message):
        """
        This function return an answer from the chatbot
        :param message:
        :return:
        """
        response = self.chat([HumanMessage(
            content=message
        )])
        return response.content

    def translateFromTo(self, message, fromLang, toLang):
        response = self.chat([HumanMessage(
            content=
            f"Translate this sentence from {fromLang} to {toLang}: {message}"
        )])
        return response.content
