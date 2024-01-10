import asyncio
import logging

import openai
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.redis import RedisStorage
from aioredis import Redis
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import sessionmaker

from bot_ai import config
from bot_ai.data.engine import get_async_engine, get_session_maker
from bot_ai.handlers import sub_pay, msg_handler, tokens_pay
from bot_ai.handlers.cb_handler import register_cb_handlers
from bot_ai.handlers.cmd_handler import register_cmd_handlers
from bot_ai.keyboards.set_menu import set_menu
from bot_ai.middlewares.mw_payment import Payment
from bot_ai.middlewares.mw_user_register import UserRegisterCheck

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    bot: Bot = Bot(token=config.BOT_TOKEN, parse_mode='HTML')
    openai.api_key = config.AI_TOKEN

    # create storage
    storage: str = config.FSM_STORAGE
    match storage:
        case 'memory':
            storage: MemoryStorage = MemoryStorage()
        case 'redis':
            redis: Redis = Redis()
            storage: RedisStorage = RedisStorage(redis)

    dp: Dispatcher = Dispatcher(storage=storage)

    # register middlewares
    dp.message.middleware(UserRegisterCheck())
    dp.callback_query.middleware(UserRegisterCheck())
    dp.callback_query.middleware(Payment())
    dp.pre_checkout_query.middleware(Payment())
    dp.message.middleware(Payment())

    # register payment
    dp.include_router(sub_pay.router)
    dp.include_router(tokens_pay.router)

    # register handlers
    register_cmd_handlers(dp)
    register_cb_handlers(dp)
    dp.include_router(msg_handler.router)  # bot_ai/handlers/msg_handler.py

    # include menu
    await set_menu(bot)

    # include postgresql
    postgresql_url: URL = URL.create(
        'postgresql+asyncpg',
        username=config.PG_USER,
        host=config.IP,
        password=config.PG_PASSWORD,
        database=config.DATABASE,
        port=int(config.PG_PORT or 0)
    )
    engine: AsyncEngine = get_async_engine(postgresql_url)
    session_maker: sessionmaker = get_session_maker(engine)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types(),
        session_maker=session_maker
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit) as exc:
        logger.error(exc)
    finally:
        logger.info("Bye!")
