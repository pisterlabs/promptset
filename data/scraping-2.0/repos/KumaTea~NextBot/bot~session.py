import logging
import configparser
from bot.store import MsgStore
from pyrogram import Client as tgClient
from openai import AsyncClient as aiClient


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

config = configparser.ConfigParser()
config.read('config.ini')

bot = tgClient(
    'rbsk',
    api_id=config['tg']['api_id'],
    api_hash=config['tg']['api_hash'],
    bot_token=config['tg']['bot_token'],
)

gpt = aiClient(
    api_key=config['openai']['api_key'],
    organization=config['openai']['organization']
)

gpt_model = config['openai']['model']
msg_store = MsgStore()
