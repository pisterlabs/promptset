from tkinter import ttk, messagebox, END
import json
import pandas as pd
import requests
import key
import openai
class ChatbotGUI:
    def __init__(self):
        # OpenAI API configuration
        openai.api_key = key.OPENAI_KEY
        self.history_user = []
        self.history_bot = []

    def initialize_chatbot(self, dataFrame):
        self.history_bot = ["I'm ready! Ask me any question about your system"]
        self.history_user = [
            "I am a user trying to understand my system and you are a computer systems professional who can help me explain my questions I have about it. Do not make it to hard to read and you should adapt for the knowledge of the user (start by supposing that the user know nothing about systems and logs). For example, if someone asks, what installations have happened in the system? You can reply with, there have been _ number of installations that have happened in the system. Examples of things you have installed include: 1. An antivirus shown from the log: Installation réussie : Windows a installé la mise à jour suivante : Mise à jour de la sélection disjointe pour Microsoft Defender Antivirus – 2267602 Ko (version 1.401.1546.0) – Canal actuel (large). You have to respond with short answers. Please use the dataset given here:"
        ]
        self.history_user.append(
            "This is the database you should analyze from:\n" + dataFrame.to_string(index=False)
            )

        self.history_bot.append(
            "Okay! I will use the database you have given me and give you my best answer!"
        )



    def add_prompt(self, user_message):
        self.history_user.append(user_message)
        messag = [{"role": "system", "content": "You are a chatbot."}]
        
        # Ajoutez l'historique de l'utilisateur sans la dernière question
        for user_message, bot_message in zip(self.history_user[:-1], self.history_bot):
            messag.append({"role": "user", "content": str(user_message)})
            messag.append({"role": "system", "content": str(bot_message)})

        # Ajoutez la dernière question de l'utilisateur
        messag.append({"role": "user", "content": str(self.history_user[-1])})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messag
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        self.history_bot.append(result)
        return result




def main():
    chatbot_gui = ChatbotGUI()
    df = chatbot_gui.jsonToArray("dataBases/system_logs_last_30_days.json")
    print("Hello, I am your Personal Virtual Technician, ask me some questions about your system and I will try my best to answer!")
    while True:
        user_input = input("Input your prompt here: ")
        if user_input == 'exit':
            break
        else:
            print("AI message:")
            print(chatbot_gui.start_query_session(user_input ,df))
        
if __name__ == "__main__":
    main()
