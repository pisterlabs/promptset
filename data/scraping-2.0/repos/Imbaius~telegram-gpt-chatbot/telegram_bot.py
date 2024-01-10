import os
import logging
from telegram import Update, constants
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
from openai import OpenAI
import json
from pydub import AudioSegment


TOKEN = open("keys/telegram_bot_key.txt", "r").read().strip("\n")
API_KEY= open("keys/openai_key.txt", "r").read().strip("\n")

if TOKEN == None:
    raise Exception("Telegram bot token is not set. Please set it in the file telegram_bot_key.txt")

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

message_history = []

def ask_chat_gpt(input_text: str):
    message_history.append({"role": "user", "content": f"{input_text}"})

    client = OpenAI(api_key=API_KEY)

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=message_history
    )

    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})

    logger.info(f"Current message history: {message_history}")

    client.close()

    return reply_content

async def chat_gpt(update: Update, context: ContextTypes.DEFAULT_TYPE):

    reply_content = ask_chat_gpt(update.message.text)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_content)

async def get_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):

    client = OpenAI(api_key=API_KEY)
    
    # get voice message and save
    new_file = await context.bot.get_file(update.message.voice.file_id)
    await new_file.download_to_drive("data/telegram.ogg")

    # convert to mp3
    recording_ogg = AudioSegment.from_ogg("data/telegram.ogg")
    recording_ogg.export("data/telegram.mp3", format="mp3")

    # read mp3 and send to openai
    recording_mp3 = open("data/telegram.mp3", "rb")
    voice_transcript = client.audio.transcriptions.create(
        file = recording_mp3, 
        model = "whisper-1",
        response_format="text"
    )

    gpt_response = ask_chat_gpt(voice_transcript)

    if gpt_response.startswith("\n\n"):
        gpt_response = gpt_response[2:]

    logger.info("GPT response: " + gpt_response)

    voice_transcript = voice_transcript

    reply_content = f"<b>{voice_transcript}</b>\n\n" + gpt_response

    logger.info(f"Reply content: {reply_content}")

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=reply_content, 
        parse_mode=constants.ParseMode.HTML
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global message_history
    message_history = []

    response = "Chat context has been reseted."
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = "Exported conversation to file"

    topic = update.message.text.replace("/export", "").strip()

    if topic != "":
        response = f"Exported conversation to file with topic: {topic}"
    else:
        topic = "no topic specified"

    message_history.append({"topic" : topic})

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

    with open('data/conversation.json', 'w', encoding='utf-8') as f:
        json.dump(message_history, f, ensure_ascii=False, indent=4)

    await reset(update, context)

    await context.bot.send_document(update.effective_chat.id, open('data/conversation.json', 'rb'))


if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)

    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_gpt))
    application.add_handler(MessageHandler(filters.VOICE , get_voice))
    application.add_handler(CommandHandler('reset', reset))
    application.add_handler(CommandHandler('export', export))
    
    application.run_polling()
