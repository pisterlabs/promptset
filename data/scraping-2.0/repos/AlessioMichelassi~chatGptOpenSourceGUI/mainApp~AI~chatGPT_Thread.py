from PyQt5.QtCore import QThread, pyqtSignal, QObject
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


class ChatGptLCThread(QThread):
    responseReceived = pyqtSignal(str)

    def __init__(self, chat_obj):
        super(ChatGptLCThread, self).__init__()
        self.chat = chat_obj
        self.message = ""

    def setMessage(self, message):
        self.message = message

    def run(self):
        # The blocking call
        response = self.chat([HumanMessage(content=self.message)])
        # Emitting signal with response content when received
        self.responseReceived.emit(response.content)


class ChatGptLC(QObject):
    temperature = 0.7
    promptTemplate = "ChatOpenAI"
    max_tokens = 800
    num_responses = 3

    answerReceived = pyqtSignal(str)

    def __init__(self):
        super(ChatGptLC, self).__init__()
        self.messageHistory = []
        self.chat = ChatOpenAI()
        # Initializing the thread and connecting its signal
        self.thread = ChatGptLCThread(self.chat)
        self.thread.responseReceived.connect(self.handleResponse)

    def handleResponse(self, content):
        self.answerReceived.emit(content)

    def getAnswer(self, message):
        """
        This function return an answer from the chatbot
        :param message:
        :return:
        """
        self.thread.setMessage(message)
        self.thread.start()
