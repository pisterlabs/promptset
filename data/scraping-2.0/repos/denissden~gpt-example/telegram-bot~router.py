from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import (
    Message
)

import openai_api

router = Router()


@router.message(Command('start'))
async def start(message: Message, state: FSMContext):
    data = await state.get_data()

    await message.answer(str(data))


@router.message(Command('reset'))
async def start(message: Message, state: FSMContext):
    await state.update_data(chat_history=None)

    await message.answer('Dialog cleared')


@router.message()
async def ai(message: Message, state: FSMContext):
    request = message.text
    data = await state.get_data()
    history = data.get("chat_history") or list()

    response = await openai_api.next_dialog_message(history, request)
    await message.answer(response)
    await state.update_data(chat_history=history)
