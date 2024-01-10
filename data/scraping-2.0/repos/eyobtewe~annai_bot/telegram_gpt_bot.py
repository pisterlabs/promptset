# Import necessary libraries
import os
from dotenv import load_dotenv
from telegram.ext import Updater, MessageHandler, Filters
from telegram import ParseMode
from moviepy.editor import AudioFileClip
import openai
import telegram


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Initialize the message history with a system message
messages = [{"role": "system", "content": "You are SuperTelegramGPT, a helpful telegram bot who is always concise and polite in its answers."}]

# Initialize the Telegram bot
updater = Updater(token=TELEGRAM_API_TOKEN, use_context=True)
dispatcher = updater.dispatcher


def text_message_handler(update, context):
    # Add user message to message history
    user_message = update.message.text
    messages.append({"role": "user", "content": user_message})

    # Use OpenAI API to generate a response based on the message history
    response = openai.Completion.create(
        model="davinci",
        prompt='\n'.join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]),
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", "Bot:"]
    )

    # Get the response message from the OpenAI API
    chatgpt_reply = response.choices[0].text.strip()

    # Send the response message back to the user
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"<b>Bot:</b> {chatgpt_reply}",
        parse_mode=ParseMode.HTML,
    )

    # Add assistant message to message history
    messages.append({"role": "assistant", "content": chatgpt_reply})
# Define the function to handle voice messages


# Define the function to handle voice messages
def voice_message(update, context):
    # Send a message to let the user know that the bot has received the voice message
    update.message.reply_text(
        "I've received a voice message! Please give me a second to respond :)")

    # Download and convert the voice message to an audio file
    voice_file = context.bot.getFile(update.message.voice.file_id)
    voice_file.download("voice_message.ogg")
    audio_clip = AudioFileClip("voice_message.ogg")
    audio_clip.write_audiofile("voice_message.mp3")

    # Use OpenAI API to transcribe the audio file
    audio_file = open("voice_message.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text

    # Send the transcription back to the user
    update.message.reply_text(
        text=f"*[You]:* _{transcript}_", parse_mode=telegram.ParseMode.MARKDOWN)

    # Add user message to message history
    messages.append({"role": "user", "content": transcript})

    # Use OpenAI API to generate a response based on the message history
    response = openai.ChatCompletion.create(
        model="text-davinci-002",
        prompt=f"Transcription of user's voice message: '{transcript}'\n\n{history_prompt(messages)}",
        temperature=0.7,
        max_tokens=1024,
        n=1,
        stop=None,
        presence_penalty=0,
        frequency_penalty=0
    )

    # Get the response message from the OpenAI API
    chatgpt_reply = response["choices"][0]["text"]

    # Send the response message back to the user
    update.message.reply_text(
        text=f"*[Bot]:* {chatgpt_reply}", parse_mode=telegram.ParseMode.MARKDOWN)

    # Add assistant message to message history
    messages.append({"role": "assistant", "content": chatgpt_reply})


# Create a Telegram bot and add message handlers
updater = Updater(TELEGRAM_API_TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(
    Filters.text & (~Filters.command), text_message_handler))

dispatcher.add_handler(MessageHandler(Filters.voice, voice_message))

# Start the bot and keep it running
updater.start_polling()
