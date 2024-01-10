from telegram.ext import Updater, MessageHandler, Filters
import openai

openai.api_key = "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"
TELEGRAM_API_TOKEN = "6623406817:AAE4HF9IR-qtM5xOS9302jn5b9ubmfAdqFY"

def text_message(update, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{"role": "system", "content": "You are a helpful assistant that always responds with a joke"}]
    )
    update.message.reply_text(response["choices"][0]["message"]["content"])


updater = Updater(TELEGRAM_API_TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), text_message))
updater.start_polling()
updater.idle()