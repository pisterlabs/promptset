import logging
import asyncio
import os

import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from django.contrib.sessions.backends.base import UpdateError

from asgiref.sync import sync_to_async
from django.core.management.base import BaseCommand
from django.conf import settings

from BotGPT.models import Dialog, Message
from bot_web_db.settings import TOKEN, TOKEN_OPENAI
from search_text.main import search_text

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())
openai.api_key = TOKEN_OPENAI


class SearchState(StatesGroup):
    awaiting_photo = State()
    searching_text = State()


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


@dp.message_handler(commands='start')
async def start_bot(message: types.Message):
    btn1 = KeyboardButton('/chat_gpt')
    btn2 = KeyboardButton('/search_text_by_photo')
    reply_mrk1 = ReplyKeyboardMarkup().add(btn1, btn2)
    await message.answer('Привет. Что ты хочешь?', reply_markup=reply_mrk1)


@dp.message_handler(commands='chat_gpt')
async def chat_text(message: types.Message):
    btn1 = KeyboardButton('/delete_dialog')
    reply_mrk2 = ReplyKeyboardMarkup().add(btn1)
    await message.answer('Скажи свой запрос \n'
                         '/delete_dialog - Удалить диалог с ассистентом', reply_markup=reply_mrk2)


@dp.message_handler(commands='search_text_by_photo')
async def text_recognition(message: types.Message):
    await message.reply('Отправь мне фотографию для распознования')
    await SearchState.awaiting_photo.set()


@dp.message_handler(content_types=types.ContentTypes.PHOTO, state=SearchState.awaiting_photo)
async def process_photo(message: types.Message, state: FSMContext):
    photo = message.photo[-1]
    file_name = f'photo{message.from_user.id}.png'
    file_path = os.path.join('static', file_name)
    await photo.download(destination_file=file_path)
    await state.update_data(photo_path=file_path)

    await message.reply('Фотография полученя. Отправьте команду /search для её распознания')
    await SearchState.searching_text.set()


@dp.message_handler(commands='search', state=SearchState.searching_text)
async def cmd_search_text(message: types.Message, state: FSMContext):
    data = await state.get_data()
    photo_path = data.get('photo_path')

    if photo_path:
        detected_text = search_text(photo_path, lang1='eng', lang2='rus')

        await message.reply(f'Распознаный текст:\n\n{detected_text}')
    else:
        await message.reply('Фото не найдено')

    await state.finish()


@dp.message_handler(commands='search_text_by_photo')
async def process_photo(message: types.Message):
    photos = message.photo


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

    btn1 = KeyboardButton('/delete_dialog')
    reply_mrk3 = ReplyKeyboardMarkup().add(btn1)

    # Получаем ответ от модели
    answer = response.choices[0].message.content

    await save_assistant_message(dialog_str, answer)

    # Отправляем ответ пользователю
    await message.answer(answer, reply_markup=reply_mrk3)
