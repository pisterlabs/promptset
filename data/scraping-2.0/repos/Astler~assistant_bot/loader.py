from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from github import Github
from pyrogram import Client

from data import config
from data.config import GITHUB_TOKEN, GITHUB_REPO, BOT_TOKEN, API_ID, API_HASH, INSTANCE_UNIQUE_NAME, OPENAI_API_KEY

bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

github = Github(GITHUB_TOKEN)
repository = github.get_user().get_repo(GITHUB_REPO)

app = Client(
    INSTANCE_UNIQUE_NAME + " pyrogram Bot App",
    bot_token=BOT_TOKEN,
    api_id=API_ID,
    api_hash=API_HASH
)

import openai

openai.api_key = OPENAI_API_KEY
