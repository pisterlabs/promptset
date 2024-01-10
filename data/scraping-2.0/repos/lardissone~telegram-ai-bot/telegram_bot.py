import logging
from uuid import uuid4
import subprocess
import os

from aiogram import Bot, Dispatcher, executor, types

from settings import OPENAI_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_IDS
from filters import IsAllowedUser
from openai_helper import transcriptor, image_generator

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("ZzzzZzZZZzz....")

@dp.message_handler(IsAllowedUser(), content_types=[types.ContentType.VOICE, types.ContentType.AUDIO])
async def transcribe(message: types.Message):
    new_msg = await message.answer("One moment, processing audio...")

    # Download file
    base_file_name = str(uuid4())
    file_name = "./tmp/" + base_file_name + ".oga"
    converted_file_name = "./tmp/" + base_file_name + ".mp3"

    if message.content_type == types.ContentType.VOICE:
        await message.voice.download(destination_file=file_name)
    else:
        await message.audio.download(destination_file=file_name)

    subprocess.run(['ffmpeg', '-y', '-i', file_name, converted_file_name])
    if os.path.isfile(file_name):
        os.remove(file_name)

    with open(converted_file_name, 'rb') as f:
        response = await transcriptor(f)

    if os.path.isfile(converted_file_name):
        os.remove(converted_file_name)

    await new_msg.edit_text(response.text)

@dp.message_handler(commands=['image'])
async def send_image(message: types.Message):
    new_msg = await message.answer("One moment, generating image...")

    prompt = message.text.replace("/image ", "")
    image_url = await image_generator(message.text.replace("/image ", ""))
    if (image_url.startswith("http")):
        await new_msg.edit_text(f'<a href="{image_url}">{prompt}</a>', parse_mode="HTML")
    else:
        await new_msg.edit_text("Error creating image.")

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
