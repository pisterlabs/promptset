import os
import telebot
from flask import Flask, request 
from pyngrok import ngrok, conf
import time
from waitress import serve
from open_ai import openAI_chatbot as openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

telegram_token = os.environ['TELEGRAM_TOKEN']
ngrok_token = os.environ['NGROK_TOKEN']
all_messages = {}

# Iniciamos el bot y el túnel
bot = telebot.TeleBot(telegram_token)
app = Flask(__name__)

# Gestionar las peticiones al servidor web POST
@app.route('/', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return 'OK', 200    

# Responder al comando /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, os.environ["TELEGRAM_START_MESSAGE"])

@bot.message_handler(content_types=['text'])
def openIA_chat(message):
    global all_messages
    chat_id = message.chat.id
    if chat_id not in all_messages:
        all_messages[chat_id] = []

    response, all_messages[chat_id] = openai.chat_with_bot(message.text, all_messages[chat_id])
    bot.send_message(chat_id, response, parse_mode="html")

@bot.message_handler(content_types=['sticker'])
def sticker_id(message):
    print(message)
    
if __name__ == "__main__":
    print("Bot iniciado")
    conf.get_default().config_path = "./ngrok.yml"
    conf.get_default().region = "sa"
    ngrok.set_auth_token(ngrok_token)
    ngrok_tunnel = ngrok.connect(5000, bind_tls=True)
    ngrok_url = ngrok_tunnel.public_url
    print(ngrok_url)
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=ngrok_url)
    serve(app, host='localhost', port=5000)
