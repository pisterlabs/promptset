from telegram import Update
from telegram.ext import ContextTypes
from helpers import db, validate_language
from openai_helper import transcribe_audio


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello! I'm Mister Said, a bot that can automatically translate messages in group chats. "
             "Please use the '/setlang [code]' command to set your preferred language. "
             "Please use only supported language codes which you can find here: https://cloud.google.com/translate/docs/languages"
    )


async def set_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_name = update.effective_user.full_name
    if len(context.args) > 0:
        lang = context.args[0]
        if validate_language(lang):
            print(f"saving language {lang} for user {user_id} in chat {chat_id}")
            doc_ref = db.collection(u'chats').document(str(chat_id)).collection(u'members').document(str(user_id))
            doc_ref.set({
                u'preferred_language': lang
            })
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"Preferred language for {user_name} is now set to {lang}")
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id,
                                           text=f"Invalid language code. Please use a supported language code, which you can find here: https://cloud.google.com/translate/docs/languages")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text=f"Please provide your two-letter language code as a parameter to the command, eg. /setlang en")


async def my_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    doc_ref = db.collection(u'chats').document(str(chat_id)).collection(u'members').document(str(user_id))
    doc = doc_ref.get()

    if doc.exists:
        user_lang = doc.to_dict()['preferred_language']
        await context.bot.send_message(chat_id=chat_id, text=f"Your current preferred language is {user_lang}.")
    else:
        await context.bot.send_message(chat_id=chat_id, text=f"You haven't set a preferred language yet. Please use the '/setlang [code]' command to set your preferred language.")


async def transcribe_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    voice = update.message.voice

    if not voice:
        return

    file_id = voice.file_id
    audio_file = await context.bot.get_file(file_id)
    audio_data = await audio_file.download_as_bytearray()

    transcription = await transcribe_audio(audio_data)

    if transcription:
        await context.bot.send_message(chat_id=chat_id, text=transcription)
    else:
        await context.bot.send_message(chat_id=chat_id, text="Failed to transcribe audio.")