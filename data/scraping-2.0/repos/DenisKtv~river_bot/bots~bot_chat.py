import logging
import os

import openai
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)


bot = Bot(token=os.getenv('CHAT_TOKEN', default='some_key'))
dp = Dispatcher(bot)
openai.api_key = os.getenv('AI_TOKEN')


async def ai(promt):
    """ Подключаем gpt-3.5-turbo и скармливаем промт нейронке, что он рыбак"""
    try:
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'Тебя зовут Петрович, ты крутой '
                 'рыбак и знаешь все о рыбалке и различных спопсобах ловли '
                 'рыбы'},
                {'role': 'user', 'content': promt}
            ]
        )
        return completion.choices[0].message.content
    except ConnectionError:
        return None


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """ Обработка команды старт"""
    await message.reply('Привет! Я эксперт в области рыбалки, что Вас '
                        'интересует?'
                        )


@dp.message_handler()
async def answer(message: types.Message):
    """ Обработка и вывод сообщений"""
    if message.text.lower() == 'рыбак эксперт':
        await message.reply('Я нейронная сеть, которая является результатом '
                            'тщательного анализа и обработки множества данных '
                            'о рыбалке, что делает ее надежным и эффективным '
                            'помощником для рыболовов, которые хотят '
                            'увеличить свой улов и улучшить свои навыки '
                            'рыбной ловли.')
    elif 'петрович' in message.text.lower():
        answer = await ai(message.text)
        if answer is not None:
            await message.reply(answer)
        else:
            logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
