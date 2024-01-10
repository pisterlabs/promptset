import json

from openai import OpenAI
from telethon import TelegramClient


class TelegramAuth:
    def __init__(self, api_id, api_hash, bot_token, openai_token):
        self.api_id = api_id
        self.api_hash = api_hash
        self.bot_token = bot_token
        self.openai_token = openai_token


def create_user_client():
    auth = load_telegram_auth()
    return TelegramClient('anon', auth.api_id, auth.api_hash)


def create_bot_client():
    auth = load_telegram_auth()
    return TelegramClient('bot', auth.api_id, auth.api_hash).start(bot_token=auth.bot_token)


def create_openai_client():
    return OpenAI(api_key=load_telegram_auth().openai_token)


telegram_auth = None


def load_telegram_auth():
    global telegram_auth
    if telegram_auth:
        return telegram_auth

    with open('token.json') as f:
        data = json.load(f)

    telegram_auth = TelegramAuth(
        api_id=data['api_id'],
        api_hash=data['api_hash'],
        bot_token=data['bot_token'],
        openai_token=data['openai_token'],
    )

    return telegram_auth
