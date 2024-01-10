import os
import openai
import asyncio
import logging
import requests
import replicate
from io import BytesIO
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.dispatcher.filters import Command

# Set the REPLICATE_API_TOKEN environment variable
# you can choose your vesrion and model here.
os.environ["REPLICATE_API_TOKEN"] = ""
model = replicate.models.get("prompthero/openjourney")
version = model.versions.get("9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb")

# some variables
START_COMMAND = """<b>Hello!</b>""" # text for /start command
CHAT_ID = -1000000000000 # your telegram chat id
DELAY = 10 #
BOT_TOKEN = '' # from BotFather
OPENAI_TOKEN = '' # token openAI

# initialization OpenAI API
openai.api_key = OPENAI_TOKEN

# bot initialization
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# logging initialization
logging.basicConfig(level=logging.INFO)

# /start
@dp.message_handler(Command('start'))
async def start_command(message: types.Message):
  await message.answer(START_COMMAND, parse_mode='HTML')

# /chat
@dp.message_handler(Command('chat'))
async def cmd_chat(message: types.Message):
  query = message.text.replace('/chat ', '')
  if len(query) > 300:
    await message.reply("your query is too big")
  else:
    response = openai.Completion.create(
      model='text-davinci-003',
		  prompt=message.text[:2042],
		  temperature=0.9,
		  max_tokens=2000,
		  top_p=1.0,
		  frequency_penalty=0.0,
		  presence_penalty=0.6,
		  stop=["You:"])
    await message.reply(response.choices[0].text)
    await asyncio.sleep(DELAY)

# /draw
@dp.message_handler(Command('draw'))
async def cmd_draw(message: types.Message):
  if message.chat.id == CHAT_ID:
    query = message.text.replace('/draw ', '')
    if len(query) > 255:
      await message.reply("your query is too big")
    else:
      response = version.predict(
        prompt=f"{query}",
        num_outputs=1,
		    image_dimensions="512x512",
		    response_format="url")
      image_url = response[0]
      image_content = requests.get(image_url).content
      image_file = BytesIO(image_content)
      await bot.send_photo(chat_id=message.chat.id, photo=image_file, caption=f'Picture: {query}')
      await asyncio.sleep(DELAY)

if __name__ == '__main__':
  executor.start_polling(dp, skip_updates=True)
