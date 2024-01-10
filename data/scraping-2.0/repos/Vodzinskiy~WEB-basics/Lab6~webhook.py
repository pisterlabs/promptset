import json
import logging
import os

import openai

from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from aiohttp import web

TOKEN = os.environ['TG_TOKEN']

WEBHOOK_HOST = 'https://vodzinskiy.alwaysdata.net/'
WEBHOOK_PATH = '/bot/'
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"
WEBAPP_HOST = '::'
WEBAPP_PORT = 8350

router = Router()
USERS_DATA_FILE = "users_data.json"
users_settings = []
error = "Please enter your OpenAI API key using command /setkey"
bot = Bot(TOKEN, parse_mode=ParseMode.HTML)


class User:
    def __init__(self, key, max_tokens, configuration):
        self.key = key
        self.max_tokens = max_tokens
        self.configuration = configuration


class Form(StatesGroup):
    key = State()
    max_tokens = State()
    configuration = State()


def load_user_settings():
    global users_settings
    try:
        with open(USERS_DATA_FILE, 'r') as f:
            try:
                users_settings = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
                users_settings = []
    except FileNotFoundError:
        logging.info("File not found, initializing with an empty list")
        users_settings = []


def get_json_data(message, key=None, configuration=None, max_tokens=None):
    load_user_settings()
    user = User(None, None, None)
    for item in users_settings:
        if item.get("id") == message.from_user.id:
            if key is not None:
                item["key"] = key
            if max_tokens is not None:
                item["max_tokens"] = max_tokens
            if configuration is not None:
                item["configuration"] = configuration
            if max_tokens is not None or configuration is not None or key is not None:
                with open(USERS_DATA_FILE, 'w') as f:
                    json.dump(users_settings, f, indent=4)
            try:
                user.key = item.get("key")
                user.configuration = item.get("configuration")
                user.max_tokens = item.get("max_tokens")
            except:
                return None
            return user


@router.message(CommandStart())
async def on_start(message: types.Message) -> None:
    load_user_settings()
    user_id = message.from_user.id

    existing_user = next((user for user in users_settings if user['id'] == user_id), None)
    if existing_user is None:
        user = {'id': user_id,
                'key': '',
                'configuration': 'You are a helpful assistant',
                'max_token': 100
                }
        users_settings.append(user)
        with open(USERS_DATA_FILE, 'w') as f:
            json.dump(users_settings, f, indent=4)


@router.message(Command("setkey"))
async def set_key(message: types.Message, state: FSMContext):
    await state.set_state(Form.key)
    await message.answer("Enter a new key")


@router.message(Command("setmaxtokens"))
async def set_max_tokens(message: types.Message, state: FSMContext):
    await state.set_state(Form.max_tokens)
    await message.answer("Enter a new max_tokens value")


@router.message(Command("setconfiguration"))
async def set_configuration(message: types.Message, state: FSMContext):
    await state.set_state(Form.configuration)
    await message.answer("Enter a new configuration value")


@router.message(Form.key)
async def key(message: types.Message, state: FSMContext) -> None:
    user = get_json_data(message, message.text, None, None)
    await message.answer(f"Key accepted: {user.key}")
    await state.clear()


@router.message(Form.max_tokens)
async def max_tokens(message: types.Message, state: FSMContext) -> None:
    user = get_json_data(message, None, None, message.text)
    await message.answer(f"Max tokens accepted: {user.max_tokens}")
    await state.clear()


@router.message(Form.configuration)
async def configuration(message: types.Message, state: FSMContext) -> None:
    user = get_json_data(message, None, message.text, None)
    await message.answer(f"Configuration accepted: {user.configuration}")
    await state.clear()


@router.message(Command("img"))
async def configuration(message: types.Message) -> None:
    user = get_json_data(message)
    if user.key is None or user.key == "":
        await message.answer(error)
    else:
        try:
            text = message.text.split('/img', 1)[1].strip()
            parts = text.split(' ', 1)
            size, prompt = parts
            if size != "256x256" and size != "512x512" and size != "1024x1024":
                await message.answer("The size is not supported, the permissible sizes are 256x256, 512x512, 1024x1024")
            else:
                try:
                    openai.api_key = user.key
                    response = openai.Image.create(
                        prompt=prompt,
                        n=1,
                        size=size
                    )
                    image_url = response['data'][0]['url']

                    await bot.send_photo(message.chat.id, image_url)
                except:
                    await message.answer("Invalid Openai key")

        except:
            await message.answer("wrong format, valid command /img <size> <prompt>")




@router.message()
async def echo(message: types.Message):
    user = get_json_data(message)
    if user.key is None or user.key == "":
        await message.answer(error)
    else:
        try:
            openai.api_key = user.key
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": user.configuration},
                {"role": "user", "content": message.text}],
                                                    max_tokens=user.max_tokens)
            await message.answer(response['choices'][0]['message']['content'])
        except:
            await message.answer("Invalid Openai key")


async def on_startup(bot: Bot) -> None:
    await bot.set_webhook(f"{WEBHOOK_URL}")


def main() -> None:
    dp = Dispatcher()
    dp.include_router(router)
    dp.startup.register(on_startup)



    app = web.Application()

    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    setup_application(app, dp, bot=bot)

    web.run_app(app, host=WEBAPP_HOST, port=WEBAPP_PORT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
