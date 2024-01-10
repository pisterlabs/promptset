from botController import BotController
from usuario import User
from dotenv import load_dotenv
import os
import telebot
from openai import OpenAI
client = OpenAI()

def salvarVoz(message, fileName):
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(fileName, 'wb') as new_file:
        new_file.write(downloaded_file)

def transcribe(fileName):
    audio_file= open(fileName, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcript


botControl = BotController()
load_dotenv()

TELEGRAM_API_KEY = os.getenv('TELEGRAM_API_KEY')
print(TELEGRAM_API_KEY)
bot = telebot.TeleBot(TELEGRAM_API_KEY, parse_mode=None) # You can set parse_mode by default. HTML or MARKDOWN

# admin = 1269325326
# users = {
# 	admin: User("Ricardo Stange", 1000, 300)
# }
users = {}




@bot.message_handler(func=lambda message: message.from_user.id not in users)
def request_access(message):
	bot.send_message(message.chat.id, "Bem vindo ao Chat Bank!")
	bot.send_message(message.chat.id, "Desenvolvido por: https://t.me/ricvs.")
	bot.send_message(message.chat.id, "Uma conta(fictícia) acaba de ser aberta em seu nome.")
	users[message.from_user.id] = User("Não implementado", -1, 500, 5000)
      
# Handles all voice files
@bot.message_handler(content_types=['voice'])
def handle_docs_audio(message):
    fileName = str(message.from_user.id)+ '.ogg'
    salvarVoz(message, fileName)
    transcript = transcribe(fileName).text
    print("Transcript: " + str(transcript))
    botResponse = botControl.processarQuery(users[message.from_user.id], transcript)
    bot.send_message(message.chat.id, botResponse)
    


@bot.message_handler(func=lambda message: message.from_user.id in users)
def send_welcome(message):
	# bot.send_message(message.chat.id, message.text)
    botResponse = botControl.processarQuery(users[message.from_user.id], message.text)
    bot.send_message(message.chat.id, botResponse)


bot.infinity_polling()


# print("Olá, sou o assistente da sua conta virtual, como posso te ajudar? ")
# while(True):
#     query = input()
#     botResponse = botControl.processarQuery(user, query)
#     print("-----> " + botResponse)
