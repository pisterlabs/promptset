import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.redis import RedisStorage
import redis.asyncio as redis
from fernet import Fernet

from utils.crypto import Crypto
from utils.openai.client import OpenAIClient
from config.settings import settings

# in code below it uses asyncio lock inside when creates connection pool
from utils.redis.redis_storage import BotChatsStorage, BotChatMessagesCache, BotOpenAIContributorChatStorage
from utils.token_api_request_manager import TokenApiRequestManager

fernet_engine = Fernet(settings.FERNET_KEY)
crypto = Crypto(fernet_engine)

redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)
storage = RedisStorage(redis) if settings.REDIS_HOST else MemoryStorage()
loop = asyncio.get_event_loop()

bot = Bot(token=settings.TG_BOT_TOKEN, parse_mode='HTML')
dp = Dispatcher(storage=storage)

bot_chats_storage = BotChatsStorage(bot.id, redis)  # To store all chats ever used by the bot.
bot_chat_messages_cache = BotChatMessagesCache(bot.id, redis, settings.TG_BOT_CACHE_TTL)
bot_contributor_chat_storage = BotOpenAIContributorChatStorage(bot.id, redis, crypto)

token_api_request_manager = TokenApiRequestManager(settings.OPENAI_TOKEN, redis, crypto)
openai_client_priority = OpenAIClient(token_api_request_manager=token_api_request_manager)
