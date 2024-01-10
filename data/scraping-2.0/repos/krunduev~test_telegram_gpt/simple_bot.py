from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
from dotenv import load_dotenv
import openai
import os
import requests
import aiohttp
import json


# подгружаем переменные окружения
load_dotenv()

# передаем секретные данные в переменные
TOKEN = os.environ.get("TG_TOKEN")
GPT_SECRET_KEY = os.environ.get("GPT_SECRET_KEY")

# передаем секретный токен chatgpt
openai.api_key = GPT_SECRET_KEY


# функция для синхронного общения с chatgpt
async def get_answer(text):
    payload = {"text":text}
    response = requests.post("http://127.0.0.1:5000/api/get_answer", json=payload)
    return response.json()


# функция для асинхронного общения с сhatgpt
async def get_answer_async(text):
    payload = {"text":text}
    async with aiohttp.ClientSession() as session:
        async with session.post('http://127.0.0.1:5000/api/get_answer_async', json=payload) as resp:
            return await resp.json()


# функция-обработчик команды /start 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # при первом запуске бота добавляем этого пользователя в словарь
    if update.message.from_user.id not in context.bot_data.keys():
        context.bot_data[update.message.from_user.id] = {'count': 3, 'history': [], 'answers': []}
    
    # возвращаем текстовое сообщение пользователю
    await update.message.reply_text('Задайте любой вопрос ChatGPT')


# функция-обработчик команды /data 
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # создаем json и сохраняем в него словарь context.bot_data
    with open('data.json', 'w') as fp:
        json.dump(context.bot_data, fp)
    
    # возвращаем текстовое сообщение пользователю
    await update.message.reply_text('Данные сгружены')

# функция-обработчик команды /data 
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    # Получение истории в виде строки
    history_text = ''
    for question, answer in zip(context.bot_data[user_id]['history'], context.bot_data[user_id]['answers']):
        history_text += f"\nВопрос: {question}\nОтвет: {answer}\n"
    await update.message.reply_text(f'Осталось запросов: {context.bot_data[user_id]["count"]}\nИстория запросов:\n{history_text}')

# функция-обработчик текстовых сообщений
async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.message.from_user.id

    # Инициализация истории сообщений для пользователя, если она еще не существует
    if user_id not in context.bot_data:
        context.bot_data[user_id] = {'count': 3, 'history': [], 'answers': []}

    # Проверка доступных запросов пользователя
    if context.bot_data[user_id]['count'] > 0:

        # Добавление вопроса в историю и поддержание истории из последних 5 вопросов
        context.bot_data[user_id]['history'].append(update.message.text)
        context.bot_data[user_id]['history'] = context.bot_data[user_id]['history'][-5:]

        # Формирование текста запроса с учетом истории
        history_text = ''
        for question, answer in zip(context.bot_data[user_id]['history'], context.bot_data[user_id]['answers']):
            history_text += f"\nВопрос: {question}\nОтвет: {answer}\n"

        # Обработка запроса пользователя
        first_message = await update.message.reply_text('Ваш запрос обрабатывается, пожалуйста подождите...')
        res = await get_answer_async( f"{update.message.text}, \n\n\n---\nИстория общения с пользователем. Используй ее для понимания контекста:\n{history_text}")
        await context.bot.edit_message_text(text=res['message'], chat_id=update.message.chat_id, message_id=first_message.message_id)

        context.bot_data[user_id]['answers'].append(res['message'])
        context.bot_data[user_id]['answers'] = context.bot_data[user_id]['answers'][-5:]

        # Уменьшение количества доступных запросов на 1
        context.bot_data[user_id]['count'] -= 1
    
    else:
        # Сообщение, если запросы исчерпаны
        await update.message.reply_text('Ваши запросы на сегодня исчерпаны')


# функция, которая будет запускаться раз в сутки для обновления доступных запросов
async def callback_daily(context: ContextTypes.DEFAULT_TYPE):

    # проверка базы пользователей
    if context.bot_data != {}:

        # проходим по всем пользователям в базе и обновляем их доступные запросы
        for key in context.bot_data:
            context.bot_data[key]['count'] = 5
        print('Запросы пользователей обновлены')
    else:
        print('Не найдено ни одного пользователя')


def main():

    # создаем приложение и передаем в него токен бота
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # создаем job_queue 
    job_queue = application.job_queue
    job_queue.run_repeating(callback_daily, # функция обновления базы запросов пользователей
                            interval=60,    # интервал запуска функции (в секундах)
                            first=10)       # первый запуск функции (через сколько секунд)

    # добавление обработчиков
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("data", data, block=False))
    application.add_handler(CommandHandler("status", status, block=False))
    application.add_handler(MessageHandler(filters.TEXT, text, block=False))

    # запуск бота (нажать Ctrl+C для остановки)
    application.run_polling()
    print('Бот остановлен')


if __name__ == "__main__":
    main()