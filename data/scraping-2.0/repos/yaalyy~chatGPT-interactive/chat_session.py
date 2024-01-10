from config import ModelSection, Temperature, Top_p, Presence_penalty, Frequency_penalty, Stop_sequences, \
    log_save_directory, api_key
# from config import Max_tokens
from openai import OpenAI
import os
from datetime import datetime


class ChatSession:

    def __init__(self, prompt=None) -> None:
        if prompt is None:
            self.__prompt = "You are a helpful assistant."
        else:
            self.__prompt = prompt
        self.__chat_history = []
        self.__model = ModelSection
        self.client = OpenAI(api_key=api_key)

    def __insertSystemPrompt(self):
        self.__chat_history.append({"role": "system", "content": self.__prompt})

    def __generateResponse(self, temp_history):
        response = self.client.chat.completions.create(
            model=self.__model,
            messages=temp_history,
            temperature=Temperature,  # default to 1
            top_p=Top_p,  # default to 1
            stop=Stop_sequences,
            presence_penalty=Presence_penalty,
            frequency_penalty=Frequency_penalty,
            # max_tokens = Max_tokens
        )
        return response

    def start(self, message=None):  # send the prompt and receive the response, user input is optional
        self.__insertSystemPrompt()
        temp_history = self.__chat_history
        if message is not None:
            temp_history.append({"role": "user", "content": message})
        response = self.__generateResponse(temp_history=temp_history)
        temp_history.append({"role": "assistant", "content": response.choices[0].message.content})
        self.__chat_history = temp_history
        return response

    def send(self, message):  # send the message from user and receive the response
        temp_history = self.__chat_history
        temp_history.append({"role": "user", "content": message})
        response = self.__generateResponse(temp_history=temp_history)
        temp_history.append({"role": "assistant", "content": response.choices[0].message.content})
        self.__chat_history = temp_history
        return response

    def clear_history(self):
        self.__chat_history.clear()

    def addToHistory(self, dict):  # add a piece of message into the history
        self.__chat_history.append(dict)

    def saveConversation(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fileName = os.path.join(log_save_directory, f"conversation_{timestamp}.txt")

        with open(fileName, "w") as file:
            for line in self.__chat_history:
                dict_string = ",".join([f"{key}: {value}" for key, value in line.items()])
                file.write(f"{dict_string}\n")

        return fileName

    def getHistory(self):
        return self.__chat_history

    def setHistory(self, list):
        self.__chat_history = list
