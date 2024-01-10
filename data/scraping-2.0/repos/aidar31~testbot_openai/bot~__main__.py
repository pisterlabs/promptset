import os 
import asyncio
import logging
from dotenv import load_dotenv, find_dotenv

from openai import OpenAI
from aiogram import Bot, Dispatcher
from aiogram.methods import DeleteWebhook

from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine 
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from db import BaseModel, User
from commands import routers
from middlewares import RegisterCheck

load_dotenv(load_dotenv())

TOKEN = os.environ.get("BOT_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
SQLALCHEMY_URL = os.environ.get("SQLALCHEMY_URL")

async def main() -> None:

    engine = create_async_engine(url=SQLALCHEMY_URL, echo=True, pool_pre_ping=True)

    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)

    session_maker = sessionmaker(engine, class_=AsyncSession)

    bot = Bot(TOKEN)
    dp = Dispatcher()
    openai_client = OpenAI(api_key=OPENAI_KEY)
    
    """ Register routers """
    for router in routers:
        dp.include_router(router)

    """ Register middlewares """
    dp.message.middleware.register(RegisterCheck())
    
    await dp.start_polling(bot, session_maker=session_maker, user_model=User, openai_client=openai_client)
    
    
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped!!!")