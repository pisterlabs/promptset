from dotenv import load_dotenv
import os 
from aiogram import Bot,Dispatcher,executor,types
import openai
import sys 

class Reference:
     '''
     A class to store previousy response from the chatgpt api 
     '''

     def __init__(self)->None:
          self.response= ""


load_dotenv()
openai.api_key=os.getenv("OpenAI_API_KEY")


reference=Reference()

TOKEN=os.getenv("TOKEN")

##Model name 
MODEL_NAME="gpt-3.5-turbo"

bot=Bot(token=TOKEN)
dp=Dispatcher(bot)


def clear_past():
     '''Clear the previous input '''
     reference.response=""


@dp.message_handler(commands=['start'])
async def welcome(message: types.Message):
    "This handler receives message with '/start" 'or' '/help'
    await message.reply("Hi\n I am TeleBot!\n Created by Vikalp(Zhcet).\nHow can i assist you?")



@dp.message_handler(commands=['clear'])
async def clear(message:types.Message):
     clear_past()
     await message.reply("I've cleared the past conversation and context ")

@dp.message_handler(commands=['help'])
async def helper(message:types.Message):
     help_command="""
     Hi There,I'm chatGPT Telegram bot created by Vikalp! Please follow these commands!!\n
     /start- to start the conversation.
     /clear - to clear the past conversation and context.
     /help - to get this help menu.
     I hope this helps. :)
      """
     await message.reply(help_command)

# @dp.message_handler()
# async def chatgpt(message:types.Message):
#      """A handler to process the users input and generate a response usng the Chatgot API """
#      print(f">>>USER:\n\t{message.text}")
#      response=openai.ChatCompletion.create(
#           model=MODEL_NAME,
#           message=[
#                {"role":"assistant","content":reference.response},
#                {"role":"user","content":message.text}
#           ]
#      )
#      reference.response=response['choices'][0]['message']['content']
#      print(f">>> Chatgpt:\n\t{reference.response}")
#      await bot.send_message(chat_id=message.chat.id,text=reference.response)
@dp.message_handler()
async def echo(message: types.Message):
    """This will return echo"""
    await message.answer(message.text)

if __name__=="__main__":
    executor.start_polling(dp,skip_updates=False)



