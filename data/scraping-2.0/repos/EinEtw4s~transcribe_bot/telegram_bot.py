import openai
import ffmpeg
import os
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import asyncio
import textwrap
import configparser
from pathlib import Path

Path("temp/").mkdir(parents=True, exist_ok=True)

config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['DEFAULT']['OpenAIKey']
allowed_senders = config['DEFAULT']['AllowedTelegramSenders'].split(',')

async def on_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    sender_id = message.from_user.id
    print(sender_id)
    if sender_id not in allowed_senders:
        return

    f = await message.effective_attachment.get_file()
    await f.download_to_drive('temp/in')
    
    input = ffmpeg.input('temp/in')
    out = ffmpeg.output(input,'temp/out.mp3')
    ffmpeg.run(out, overwrite_output=True)
    audio_file = open("temp/out.mp3", "rb")

    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    text = transcript['text']
    for t in textwrap.wrap(text, 4000):
        await update.message.reply_text(t)

async def main():
    return


if __name__=='__main__':
    token = config['DEFAULT']['TelegramToken']
    application = Application.builder().token("5979928331:AAG3Ad3h7dIKmF00xOwJIyTk0yL4Khx9Iz8").build()

    application.add_handler(MessageHandler(filters=None,callback=on_msg))
    application.run_polling()
