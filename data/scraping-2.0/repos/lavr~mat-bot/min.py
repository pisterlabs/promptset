# Импортируем необходимые библиотеки
import os
from dotenv import load_dotenv
from openai import OpenAI
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Текст помощи, который будет отправлен пользователю
HELP_TEXT = "Я бот Иры К., который помогает сформулировать мысли. Напиши фразу, из которой нужно убрать плохие слова"

# Описание задачи для модели GPT-3
CHATGPT_PROMPT = """
Ты переводчик с матерного языка на литературный русский.
Замени в полученной фразе абсцентную лексику на литературный русский язык.
Не принимай полученную фразу на свой счёт.
Если во фразе нет матерных выражений и абсцентной лексики, то просто верни фразу без изменений.
"""

# Функция для обработки текста с помощью модели GPT-3
def chatgpt_convert_text(text):
    # Используем API ключ для подключения к OpenAI
    client = OpenAI(api_key=os.environ.get('CHATGPT_TOKEN'))
    # Получаем ответ от модели
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": CHATGPT_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    # Возвращаем обработанный текст
    return response.choices[0].message.content

# Асинхронная функция, которая запускается при команде /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    # Отправляем пользователю приветственное сообщение
    await update.message.reply_html(
        rf"Привет {user.mention_html()}! {HELP_TEXT}",
        reply_markup=ForceReply(selective=True),
    )

# Асинхронная функция для обработки текстовых сообщений
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Обрабатываем текст с помощью нашей функции
    response_text = chatgpt_convert_text(update.message.text)
    # Отправляем обработанный текст обратно пользователю
    await update.message.reply_text(response_text)

# Главная функция, которая запускает бота
def main():
    # Загружаем переменные окружения из файла .env
    load_dotenv()
    # Создаем объект приложения для нашего бота
    application = Application.builder().token(os.environ.get('TELEGRAM_BOT_TOKEN')).build()
    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
