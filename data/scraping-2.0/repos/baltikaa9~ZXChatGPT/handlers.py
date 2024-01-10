import asyncio

from aiogram import Dispatcher
from aiogram.types import Message
from openai.error import RateLimitError

from create_bot import bot
from decorators import is_private_chat
from services import get_chat_gpt_response, set_context, get_context


async def start(message: Message):
    await message.answer('Привет! Я - ChatGPT, искусственный интеллект, разработанный компанией OpenAI. Моя основная '
                         'задача - помогать отвечать на вопросы и предоставлять информацию по различным темам. Я '
                         'основан на технологии глубокого обучения и обучен на большом объеме текстовых данных, '
                         'что позволяет мне предоставлять ответы и выполнять различные задачи. Пожалуйста, '
                         'задайте свой вопрос, и я постараюсь вам помочь!')


async def chat_gpt_answer(message: Message):
    if message.text.startswith('/q'):
        try:
            message.text = message.text.split(maxsplit=1)[1]
        except IndexError:
            ...

    bot_message = await message.answer('Печатает...')

    try:
        chat_gpt_message = await get_chat_gpt_response(message.text, message.chat.id)
    except RateLimitError:
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=bot_message.message_id,
        )
        await message.reply('Пожалуйста подождите 20 секунд.')
        await asyncio.sleep(20)
        await message.reply('Можете задавать свой вопрос.')
    else:
        await bot.edit_message_text(
            message_id=bot_message.message_id,
            text=chat_gpt_message,
            chat_id=message.chat.id,
            parse_mode='markdown'
        )


async def get_chat_context(message: Message):
    context = get_context(message.chat.id)
    context = [item.get('content') for item in context]
    if context:
        await message.answer('● ' + '\n● '.join(context))
    else:
        await message.answer('История пуста')


async def delete_context(message: Message):
    set_context(message.chat.id, [])
    await message.answer('Контекст очищен')


def register_handlers(dispatcher: Dispatcher):
    dispatcher.register_message_handler(start, commands=['start'])
    dispatcher.register_message_handler(get_chat_context, commands=['context'])
    dispatcher.register_message_handler(delete_context, commands=['deletecontext'])
    dispatcher.register_message_handler(chat_gpt_answer, commands=['q'])
    dispatcher.register_message_handler(is_private_chat(chat_gpt_answer), content_types='text')
