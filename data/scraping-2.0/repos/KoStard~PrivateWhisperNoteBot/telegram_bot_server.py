import os
import uuid

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import dotenv
import logging
from openai import OpenAI
import io
from pydub import AudioSegment

import json

from message_texts import get_welcome_message, get_access_denied_message, get_help_message

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

dotenv.load_dotenv()
client = OpenAI()

ALLOWED_USERS = json.loads(os.getenv("ALLOWED_USERS"))


def with_exception_replying(message):
    def inner(func):
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            try:
                await func(update, context)
            except Exception as e:
                logging.error(e)
                await update.effective_message.reply_text(message, quote=True)

        return wrapper
    return inner


async def authenticate(update: Update):
    if update.effective_user.id not in ALLOWED_USERS:
        await update.effective_message.reply_text(get_access_denied_message())
        raise Exception(
            f"User {update.effective_user.username} with id {update.effective_user.id} tried to use the bot.")


async def handle_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await authenticate(update)
    response_message = get_welcome_message(update.effective_user.first_name)
    await update.message.reply_text(response_message)


async def handle_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response_message = get_help_message()
    await update.message.reply_text(response_message)


@with_exception_replying("Something went wrong with the processing of this file")
async def transcribe_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await authenticate(update)
    file = await update.message.voice.get_file()
    await _handle_audio_file(update, file)


@with_exception_replying("Something went wrong with the processing of this file")
async def transcribe_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await authenticate(update)
    file = await update.message.audio.get_file()
    await _handle_audio_file(update, file)


async def _handle_audio_file(update, file):
    """
    We are converting the received audio file to mp3 file, as sometimes some audio files are not accepted by whisper,
    and it works better after converting to mp3. This adds some latency, so we can tackle to receive some optimization.
    """
    file_format = file.file_path.split(".")[-1]
    path = await file.download_to_drive(f"/tmp/{uuid.uuid4()}.{file_format}")
    with open(path, 'rb') as downloaded_file:
        converted_audio = _convert_audio(downloaded_file, file_format)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.mp3", converted_audio.read())
        )
        await update.effective_message.reply_text(transcript.text)


def _convert_audio(input_file, file_format) -> io.BytesIO:
    logging.info("Converting the audio file")
    mp3_io = io.BytesIO()
    AudioSegment.from_file(input_file, type=file_format).export(mp3_io, format="mp3", bitrate="64")
    mp3_io.seek(0)
    logging.info("Done converting the audio file")
    return mp3_io


async def invalid_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("This seems to be an invalid command. Please enter / to see the available "
                                    "commands or use /help for more information about the bot.")


app = ApplicationBuilder().token(os.environ['TELEGRAM_BOT_API_KEY']).build()

app.add_handler(CommandHandler("start", handle_start_command))
app.add_handler(CommandHandler("help", handle_help_command))
app.add_handler(MessageHandler(filters.VOICE, transcribe_voice))
app.add_handler(MessageHandler(filters.AUDIO, transcribe_audio))
app.add_handler(MessageHandler(filters.COMMAND, invalid_command_handler))
app.run_polling()
