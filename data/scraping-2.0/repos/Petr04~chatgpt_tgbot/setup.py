from settings import (
  BOT_TOKEN,
  CHATGPT_ORGANIZATION, CHATGPT_TOKEN,
  REDIS, DB_URL,
)
from tortoise import Tortoise, run_async
from aiogram import Bot, Dispatcher
# from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.fsm_storage.redis import RedisStorage2
import openai


bot = Bot(BOT_TOKEN)
Bot.set_current(bot)
# storage = MemoryStorage()
storage = RedisStorage2(**REDIS)
dp = Dispatcher(bot, storage=storage)
Dispatcher.set_current(dp)


async def db_init():
  await Tortoise.init(
    # db_url='sqlite://db.sqlite3',
    db_url=DB_URL,
    modules={'models': ['models']}
  )
  await Tortoise.generate_schemas()

run_async(db_init())


openai.organization = CHATGPT_ORGANIZATION
openai.api_key = CHATGPT_TOKEN
