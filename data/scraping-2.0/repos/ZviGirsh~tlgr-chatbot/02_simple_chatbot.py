from telegram.ext import Updater, MessageHandler, Filters
import telegram
import openai

openai.api_key = "<key>"
TELEGRAM_API_TOKEN = "<key>"
model = "gpt-3.5-turbo-16k"

messages = [{"role": "system", "content": "You are TelegramGPT, a helpful telegram bot that is always concise and polite in its answers."}]


def text_message(update, context):
    messages.append({"role": "user", "content": update.message.text})
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    update.message.reply_text(text=f"*[Bot]:* {ChatGPT_reply}", parse_mode=telegram.ParseMode.MARKDOWN)
    messages.append({"role": "assistant", "content": ChatGPT_reply})


updater = Updater(TELEGRAM_API_TOKEN, use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), text_message))
updater.start_polling()
updater.idle()