import os
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ContextTypes

load_dotenv()
'''Загружаем Ключи Доступа'''
TOKEN = os.getenv("GIGA_CHAT_TOKEN")

async def ai_help(update:Update, context:ContextTypes.DEFAULT_TYPE):
    '''Функция, которая вызывает помощник по искусственному интеллекту с помощью ключа доступа'''
    chat = GigaChat(
        credentials=TOKEN,
        verify_ssl_certs=False)

    messages = [
        SystemMessage(
            content="Ты мужской эмпатичный бот-психолог-помощник по именем Аади который помогает пользователю решить его проблемы."

        )
    ]

    '''Получение текста от пользователя и отправка ответа от помощника пользователю'''
    user_input = update.message.text
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Помощник:\n{res.content}")

