import json
import os
from dotenv import load_dotenv
import telebot
import sqlite3
import openai
import requests

load_dotenv()

TOKEN = os.getenv('TOKEN')
AI_TOKEN = os.getenv('AI_TOKEN')

bot = telebot.TeleBot(TOKEN)


# openai.api_key = AI_TOKEN  # supply your API key however you choose

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Добро пожаловать!")


mass_messages = []


@bot.message_handler(func=lambda message: True)
def main(message):
    mass_messages.append({"role": "user", "content": message.text})
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_TOKEN}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": mass_messages
    }

    response = requests.post(url, headers=headers, json=data)
    print(response.text)
    assistant_answer = json.loads(response.text)['choices'][0]['message']['content']

    mass_messages.append({"role": "assistant", "content": assistant_answer})
    assistant_answer = assistant_answer.replace('```', '\n ``` \n')
    bot.send_message(message.chat.id, assistant_answer)


if __name__ == "__main__":
    print("Start bot")
    bot.infinity_polling()
