import config

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat


router = Router()


@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(config.START_MESSAGE)


@router.message()
async def message_handler(msg: Message):
    chat = GigaChat(credentials=config.GIGA_CHAT_AUT, verify_ssl_certs=False)
    messages = [
        SystemMessage(content=config.START_MESSAGE),
        HumanMessage(content=msg.text)
    ]
    res = chat(messages)

    await msg.answer(f"{res.content}")