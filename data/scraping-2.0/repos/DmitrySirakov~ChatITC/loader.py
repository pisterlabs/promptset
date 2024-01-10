"""
Loader module for initializing the bot and loading necessary data.
"""
import asyncio
from aiogram import Bot, Dispatcher
import openai
from config import TOKEN, OPENAI_API_KEY
from services.user_service import (
    load_users_from_google_sheets,
    load_admins_from_google_sheets
)

# Initialize the bot, dispatcher, and OpenAI API key
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
openai.api_key = OPENAI_API_KEY

# Load users and admins before the bot starts.
loop = asyncio.get_event_loop()
loop.run_until_complete(load_users_from_google_sheets('dppcommands-7a27921d2259.json', 'Верификация GPT ITC'))
loop.run_until_complete(load_admins_from_google_sheets('dppcommands-7a27921d2259.json', 'Админы GPT ITC'))
