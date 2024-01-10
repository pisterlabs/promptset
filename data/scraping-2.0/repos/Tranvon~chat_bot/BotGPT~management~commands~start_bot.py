import logging
import asyncio
import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from django.contrib.sessions.backends.base import UpdateError

from bot_web_db.settings import TOKEN, OPENAI_TOKEN

from asgiref.sync import sync_to_async
from django.core.management.base import BaseCommand
from django.conf import settings

from BotGPT.models import Dialog, Message

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
openai.api_key = OPENAI_TOKEN

# Настройка логгера
logging.basicConfig(level=logging.ERROR)
print('Start Bot!')


class Command(BaseCommand):
    help = 'Telegram bot setup command'

    def handle(self, *args, **options):
        sync_to_async(executor.start_polling(dp, skip_updates=True))


# Настройка логгера
logging.basicConfig(level=logging.ERROR)

# Создание объекта логгера
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)

# Создание файлового обработчика
file_handler = logging.FileHandler('error.log')

# Установка формата записи в файле
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Добавление обработчика в логгер
logger.addHandler(file_handler)


# Обработчик исключений для бота
@dp.errors_handler()
async def error_handler(update, exception):
    # Логирование исключения в файл
    logger.error(f"An error occurred: {exception}")
    if str(exception).find('Please try again in 20s') != -1:
        # Отправка сообщения об ошибке в чат или выполнение других действий
        await update.message.reply("Я не такой быстрый, столько много сообщений не приму!.")
    else:
        # Отправка сообщения об ошибке в чат или выполнение других действий
        await update.message.reply("Произошла ошибка. Пожалуйста, повторите позже.")


# Обработчик ошибки асинхронных операций
@dp.errors_handler(exception=asyncio.TimeoutError)
async def timeout_error_handler(update, exception):
    # Логирование ошибки
    logger.error(f"Timeout error occurred: {exception}")

    # Отправка сообщения об ошибке в чат или выполнение других действий
    await update.message.reply("Истекло время ожидания. Пожалуйста, повторите позже.")


# Обработчик ошибки сетевого подключения
@dp.errors_handler(exception=ConnectionError)
async def connection_error_handler(update, exception):
    # Логирование ошибки
    logger.error(f"Connection error occurred: {exception}")

    # Отправка сообщения об ошибке в чат или выполнение других действий
    await update.message.reply("Ошибка подключения. Пожалуйста, повторите позже.")


# Обработчик ошибки обновления
@dp.errors_handler(exception=UpdateError)
async def update_error_handler(update, exception):
    # Логирование ошибки
    logger.error(f"Update error occurred: {exception}")

    # Отправка сообщения об ошибке в чат или выполнение других действий
    await update.message.reply("Ошибка обновления. Пожалуйста, повторите позже.")


@sync_to_async
def save_user_message(dialog, user_input):
    role_user = "user"
    dialog_obj, _ = Dialog.objects.get_or_create(username=f"{dialog}", role=role_user)
    user_message = Message(dialog=dialog_obj, role=role_user, content=user_input)
    user_message.save()


@sync_to_async
def save_assistant_message(dialog, answer):
    role_assistant = "assistant"
    dialog_obj, _ = Dialog.objects.get_or_create(username=f"{dialog}", role=role_assistant)
    assistant_message = Message(dialog=dialog_obj, role=role_assistant, content=answer)
    assistant_message.save()


@dp.message_handler(commands=['delete_dialog'])
async def delete_dialog(message: types.Message):
    dialog_str = f"{message.from_user.username}"

    # Получаем диалоги, которые нужно удалить
    dialogs = await sync_to_async(Dialog.objects.filter)(username=dialog_str)

    # Преобразуем асинхронный QuerySet в синхронный список
    dialogs = await sync_to_async(list)(dialogs)

    # Удаляем каждый диалог с помощью синхронного вызова delete()
    for dialog in dialogs:
        await sync_to_async(dialog.delete)()

    # Получаем сообщения, связанные с удаленными диалогами
    messages = await sync_to_async(Message.objects.filter)(dialog__username=dialog_str)

    # Преобразуем асинхронный QuerySet в синхронный список
    messages = await sync_to_async(list)(messages)

    # Удаляем каждое сообщение с помощью синхронного вызова delete()
    for message in messages:
        await sync_to_async(message.delete)()

    await message.reply("Диалог с ассистентом удален.")


@dp.message_handler()
async def handle_message(message: types.Message):
    if message.text == "/delete_dialog":
        await delete_dialog(message)

    user_input = message.text

    dialog_str = f"{message.from_user.username}"

    await save_user_message(dialog_str, user_input)

    # Получаем предыдущие сообщения диалога из базы данных
    dialog_objs = await sync_to_async(Dialog.objects.filter)(username=f"{dialog_str}")
    previous_messages = await sync_to_async(Message.objects.filter)(dialog__in=dialog_objs)

    # Формируем список сообщений для запроса к модели OpenAI
    messages = await sync_to_async(
        lambda: [
                    {"role": "system", "content": "You are a helpful assistant"},
                ] + [
                    {"role": message.role, "content": message.content}
                    for message in previous_messages
                ] + [
                    {"role": "user", "content": user_input}
                ]
    )()

    # Отправляем запрос на модель GPT-3.5 Turbo с полным диалогом
    response = await sync_to_async(openai.ChatCompletion.create)(
        model="gpt-3.5-turbo-0301",
        messages=messages
    )

    # Получаем ответ от модели
    answer = response.choices[0].message.content

    await save_assistant_message(dialog_str, answer)

    # Отправляем ответ пользователю
    await message.answer(answer)
