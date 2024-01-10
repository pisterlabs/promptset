import os
import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes
import openai
import logger

openai.api_key = os.getenv("OPEN_AI_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def log_response(prompt, response):
    logger.log(prompt, response)


async def chatgpt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Voce se chama White Pixel. Você é uma assistente útil e concisa."},
            {"role": "user", "content": update.message.text}
        ]
    )

    if response['choices'][0]['finish_reason'] != 'stop':
        print(response)
        return

    print(response['choices'][0]['message']['content'])
    log_response(update.message.text, response['choices'][0]['message']['content'])

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=response['choices'][0]['message']['content']
    )


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")


if __name__ == '__main__':
    application = ApplicationBuilder().token(os.environ['TELEGRAM_TOKEN']).build()

    chatgpt_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chatgpt)

    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application.add_handler(chatgpt_handler)
    application.add_handler(unknown_handler)

    application.run_polling()
