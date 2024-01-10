from dotenv import load_dotenv
import os
from aiogram import Bot, Dispatcher, executor,types
import openai
import sys

class Reference:
    '''
    A Class to store previously response from the chatGPT API
    '''

    def __init__(self) -> None:
        self.response = ""


load_dotenv()
openai.api_key = os.getenv("OpenAI_API_KEY")

reference = Reference()

TOKEN = os.getenv("TOKEN")

MODEL_NAME = "gpt-3.5-turbo"

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

def clear_past():
    reference.response = ""

@dp.message_handler(commands=['start','help'])
async def welcome(message:types.Message):
    await message.reply("Hi\nI am Tele Bot!\Created by Santhosh. How can I assist you?")

@dp.message_handler(commands=['clear'])
async def welcome(message:types.Message):
    """
    A Handler to clear the previous conversation and context
    """
    clear_past()
    await message.reply("I've cleared the past conversation and context.")

@dp.message_handler(commands=['help'])
async def helper(message: types.Message):
    help_command = """
             Hi There, I'm chatGPT Telegram bot created by Santhosh! Please follow these command - 
             /start - to start the conversation
             /clear - to clear the past conversation and context.
             /help - to get this help menu.
             I hope this helps. :)"""
    await message.reply(help_command)

@dp.message_handler()
async def chatgpt(message: types.Message):
    print(f">>> USER: \n\t{message.text}")
    response = openai.ChatCompletion.create(
        model = MODEL_NAME,
        message = [
            {"role":"assistant","content":reference.response},
            {"role":"user","content":message.text}
        ]

    )
    reference.response = response['choices'][0]['message']['content']
    print(f">>> chatGPT: \n\t{reference.response}")
    await bot.send_message(chat_id=message.chat.id, text=reference.response)

if __name__=="__main__":
    executor.start_polling(dp,skip_updates=True)
