# coding=utf-8
import openai
import os
import telebot

BOT_TOKEN = os.environ.get('BOT_TOKEN')

OPENAI_API_TOKEN = os.environ.get('OPENAI_API_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)

openai.api_key = OPENAI_API_TOKEN

info = []

@bot.message_handler(func=lambda msg: True)

def echo_all(message):
    global info
    if(message.text == "stopChat" or message.text=="停止当前对话" or message.text=="停止对话"):
        info = []
        try:
            bot.reply_to(message, "对话已停止，谢谢。")
        except Exception as e:
            print(e)
            bot.reply_to(message, "telegramBot遇到了问题，请重试。")
    else:
        try:
            info.append({"role": "user", "content": message.text}) 
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=info)
            info.append(completion.choices[0].message) 
            answer = completion.choices[0].message["content"]
        except Exception as e:
            answer = "chatGPT遇到了问题，请重试。"
            print(e)
   
        try:
            bot.reply_to(message, answer)
        except Exception as e:
            print(e)
            bot.reply_to(message, "telegramBot遇到了问题，请重试。")

bot.infinity_polling()
