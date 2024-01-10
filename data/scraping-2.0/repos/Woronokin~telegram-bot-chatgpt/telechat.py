import openai
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor


token = 'TELEGRAM_BOT_TOKEN'
openai.api_key = 'OPENAI_KEY'

bot = Bot(token)
dp = Dispatcher(bot)

# print(openai.Model.list())

users = {tg_id1,tg_id2}

accepted_users = lambda message: message.from_user.id not in users

@dp.message_handler(accepted_users, content_types=['any'])
async def handle_unwanted_users(message: types.Message):
    await message.answer("Извините, бот работает только для одобренных пользователей. Если вы хотите написать такой же бот - перейдите по ссылке: https://nikonorow.ru/pishem-telegram-bota-chatgpt-na-python/")
    return

max_symbols = lambda message: int(len(message.text)) > 2000

@dp.message_handler(max_symbols, content_types=['any'])
async def handle_unwanted_users(message: types.Message):
    await message.answer("Ошибка! Введенное количество символов превышает максимальное значение в 2000 \n\nКоличество введенных символов: " + str(len(message.text)) + "\n\nСократите Ваш запрос \n\nЧитайте подробнее о токенах OpenAI API: https://nikonorow.ru/kolichestvo-tokenov-postupleniya-openai-api-python/")
    return

@dp.message_handler()
async def send(message: types.Message):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=message.text,
        temperature=0.9,
        max_tokens=4000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    await message.answer(response['choices'][0]['text'])

executor.start_polling(dp, skip_updates=True)


