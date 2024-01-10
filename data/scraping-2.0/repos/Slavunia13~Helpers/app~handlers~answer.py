from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters.callback_data import CallbackData

import requests
import json

from datetime import datetime, timedelta

import openai
from random import randint
import calendar

router = Router()

users_answer = {}



@router.message(F.text.startswith("Задать свой вопрос"))
async def notited(message: Message):
    global users_answer
    users_answer[message.from_user.id] = 1
    await message.answer("Задавайте свой вопрос:")

@router.callback_query(F.data == "add_user")
async def next_month(callback: CallbackQuery):
    global users_answer
    users_answer[callback.from_user.id] = 1

    await callback.message.answer("Задавайте свой вопрос:")


@router.message(F.text)
async def keep_answer(message: Message):
    if(not message.from_user.id in users_answer):
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(
            text="Да, я хочу",
            callback_data="add_user")
        )

        await message.answer("Вы хотите задать вопрос?", reply_markup=builder.as_markup())
    else:
        await message.answer("Вопрос был отправлен..")
        promt = str(message.text)
        print(f"{promt=}")

        openai.api_key = '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'

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": "Представь, что ты профессиональный психолог и психотерапевт, мне нужна помощь. Подскажи ответ на вопрос на руссмком - " + promt,
                },
            ],
        )

        await message.answer(completion.choices[0].message.content)


