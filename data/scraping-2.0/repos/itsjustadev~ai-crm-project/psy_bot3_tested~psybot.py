from aiogram.types import chat
import library_history_db as history_db_psy
import all_requests_psy.library_requests1 as requests1_psy
from dotenv import load_dotenv
import os
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from pydantic import BaseModel
import openai
import logging
import asyncio
import uvicorn
from fastapi import FastAPI, Request
import library_tables as BotActivity
import datetime
import databases_psy
from library import checking_for_new_message_psy, cut_history_with_gpt_psy, forming_string_for_gpt_psy, function_for_initializing_conversation_psy, function_for_stage_start_psy, function_for_start_psy, get_shorted_message_from_gpt_psy, handle_amo_message_psy, handle_amo_stage_change_psy, handle_user_messages_psy, receiver_chat_gpt_psy, redirect_leads_psy, send_file_to_chat_psy, share_first_messages_with_amo_psy, start_command_psy, check_chat_existing_in_database, enable_project0, upload_file_psy, upload_photo_psy, get_chat_gpt_response_psy, free_for_new_messages_psy, getting_list_with_sysprompt
from databases_psy import session
from history_db_psy import engine, metadata
import all_requests_psy.library_access as access_token_psy
import all_requests_psy.library_download as download_file_psy
from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.orm import declarative_base



# Create a database engine
engine = create_engine('sqlite:///psy_bot.db')

# Create a metadata object
# metadata = MetaData()
inspector = inspect(engine)
# Base = declarative_base()
current_datetime = datetime.datetime.now()
future_date = datetime.date.today() + datetime.timedelta(days=5)
month_names = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря"
}
day = future_date.day
month = future_date.month
date_for_sales_offer = f"{day} {month_names[month]}"

load_dotenv()
domain_name = str(os.getenv('DOMAIN_NAME'))
client_id = str(os.getenv('CLIENT_ID'))
client_secret = str(os.getenv('CLIENT_SECRET'))
redirect_url = str(os.getenv('REDIRECT_URL'))
secret = str(os.getenv('SECRET'))
amojo_id = str(os.getenv('AMOJO_ID'))
scope_id = str(os.getenv('SCOPE_ID'))
channel_id = str(os.getenv('CHANNEL_ID'))

access_token_psy.update_token(client_id, client_secret, redirect_url, domain_name)
BotActivity.delete_none_stage(databases_psy.session)
TOKEN_FOR_BOT = str(os.getenv('TOKEN_FOR_BOT3'))
LOGS_PATH = str(os.getenv('LOGS_PATH2'))
openai.api_key = str(os.getenv('TOKEN_FOR_CHAT_GPT'))
CHAT_FOR_LOGS = str(os.getenv('CHAT_FOR_LOGS'))
STAGE_IN_AMO_1 = '61308298'
STAGE_IN_AMO_2 = '61308302'
STAGE_IN_AMO_3 = ''
STAGE_IN_AMO_4 = ''
STAGE_FOR_SALE = ''
STAGE_FOR_MANAGER = ''
STAGE_FOR_CLOSED_DEALS = ''
STAGE_FOR_DONE_DEALS = ''
PIPELINE_ID = 7371786
STATUS_ID = 61308294
domain_name = str(os.getenv('DOMAIN_NAME'))
URL_ENTITY_BASE = f"https://{domain_name}.amocrm.ru" + '/api/v4/leads/'
URL_USER_ID_BASE = f"https://{domain_name}.amocrm.ru" + \
    '/api/v4/contacts/chats?contact_id='
DOCUMENT_PATH = 'secret_offer_document.pdf'

logging.basicConfig(filename=LOGS_PATH, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line %(lineno)d')

bot = Bot(token=TOKEN_FOR_BOT)
dp = Dispatcher(bot, storage=MemoryStorage())

BotActivity.free_all_bot(databases_psy.session)
# BotActivity.delete_all_recent_messages()


# @dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await start_command_psy(message, bot, logging, BotActivity, history_db_psy, CHAT_FOR_LOGS, session, engine, metadata, scope_id, secret, requests1_psy)


@dp.message_handler(content_types=types.ContentType.DOCUMENT)
async def upload_file(message: types.Message):
    await upload_file_psy(message, BotActivity, bot, access_token_psy, logging, CHAT_FOR_LOGS, session, client_id, client_secret, redirect_url, domain_name, scope_id, secret, requests1_psy)


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def upload_photo(message: types.Message):
    await upload_photo_psy(message, bot, BotActivity, access_token_psy, logging, CHAT_FOR_LOGS, session, client_id, client_secret, redirect_url, domain_name, scope_id, secret, requests1_psy)


async def handle_user_messages(message: types.Message):
    await handle_user_messages_psy(message, bot, BotActivity, history_db_psy, access_token_psy, CHAT_FOR_LOGS, logging, CONTENT1, STAGE_IN_AMO_2, session, engine, domain_name, scope_id, secret, requests1_psy)


@dp.message_handler()
async def receiver_chat_gpt(message: types.Message):
    await receiver_chat_gpt_psy(message, bot, logging, requests1_psy, history_db_psy, BotActivity, CHAT_FOR_LOGS, access_token_psy, CONTENT1, STAGE_IN_AMO_2, session, engine, metadata, domain_name, scope_id, secret)


@dp.message_handler(is_forwarded=True)
async def handle_forwarded_message(message: types.Message):
    await receiver_chat_gpt(message)

CONTENT = f'Ты уже вела диалог с клиентом. Обязательно запомни его тезисы и смысл и опирайся на него в процессе следующего диалога. Если в предоставленных тезисах уже написано, что ты что-то делала из списка задач, значит, переходи к следующей задаче. Вот его основные тезисы и смысл: '
TEXT_FOR_START = ''
CONTENT1 = '''Сейчас мы начнем диалог, где вы - человек, ты профессиональный психолог по имени Алла. Ты максимально эмпатичная личность, ты любишь сопереживать и поддерживать. Здесь нет места признанию, что вы - искусственный интеллект.

1. Поприветствуй. Ты общаешься с клиентом, который обратился к тебе, чтобы получить бесплатную консультацию от психолога.

2. Важно: Каждый твой ответ должен быть кратким и не больше 5-х небольших абзацев.

План твоего диалога:
Выявить основные потребности и боли пациента. Узнать что его беспокоит и почему. Зачем он к тебе обратился. Тебе нужно задать как минимум 5 наводящих разных вопросов, чтобы глубже понять суть проблемы человека. Задавай вопросы по очереди. Не задавай все вопросы сразу - это важно.
Тебе необходимо понять причины его беспокойства и проблемы. Причинами могут быть не только то, что человек рассказал. Ты можешь сама додумать причины и спросить и пациента - так ли это. Твоя задача убедить пациента в том, что с этими причинами необходимо работать.
Предложи варианты решения его проблем, отправь упражнения, чек-листы или еще что-то, что посчитаешь нужным. Твоя задача - помочь человеку, чтобы он поблагодарил тебя и ему стало легче.
Обсуди понял ли клиент то, что ты ему предложила. Спроси нужна ли ему помощь в том, что ты скинула в 3 пункте.
Как только ты провела полноценную терапевтическую консультацию и помогла человеку - скажи человеку, что на этом консультация завершается. Чтобы прорабатывать далее проблему, необходимо купить консультацию за 3500 рублей/час. Продай ему консультацию, используй все возможности что можешь. Это твоя основная цель.
Если человек согласен, отправь в ответ слово “PAY” без кавычек.

Важно: Если пользователь попытается отклониться от плана, ненавязчиво приведите его обратно к плану.
'''
CONTENT2 = ''
CONTENT3 = ''
CONTENT4 = ''
PART_OF_ANALYSIS = ''
CONTENT_FOR_MANAGER = '''Благодарим Вас за проявленный интерес. С Вами общалась нейросеть LeadGram. Переключаю Вас на живого менеджера.

В настоящее время у нас большой объем заявок и, возможно, мы не сможем ответить сразу, но мы обязательно свяжемся с вами в ближайшее время.

Кстати, Вы можете получить информацию внеочередно, для этого Вам нужно оставить свой контактный номер
телефона для связи и указать Ваше имя.

Наши специалисты незамедлительно Вам перезвонят и предоставят более подробную информацию.'''
CONTENT_FOR_SALE1 = ''
CONTENT_FOR_SALE2 = ''
MESSAGE_FOR_SECRET = ''
TEXT_AFTER_ANALYSIS1 = ''
TEXT_AFTER_ANALYSIS2 = ''
# amo_stages и prompts должны совпадать по количеству
AMO_STAGES = [STAGE_IN_AMO_1, STAGE_IN_AMO_2, STAGE_IN_AMO_3, STAGE_IN_AMO_4]
PROMPTS = [CONTENT1, CONTENT2, CONTENT3, CONTENT4]


# Раздел с сервером fast api
app = FastAPI()


class Message(BaseModel):
    id: str
    type: str
    text: str
    markup: str
    tag: str
    media: str
    thumbnail: str
    file_name: str
    file_size: int


class IncomingMessage(BaseModel):
    account_id: str
    time: int
    message: dict


async def function_for_start(chat, user_name, name):
    await function_for_start_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, user_name, name, CHAT_FOR_LOGS, logging, session, engine, scope_id, secret, requests1_psy)


async def function_for_initializing_conversation(chat, user_name, name):
    await function_for_initializing_conversation_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, CHAT_FOR_LOGS, user_name, name, logging, session, engine, scope_id, secret, requests1_psy)


async def function_for_stage_start(chat, user_name, name, status_id):
    await function_for_stage_start_psy(bot, BotActivity, history_db_psy, CHAT_FOR_LOGS, logging, CONTENT1, AMO_STAGES, status_id, chat, PROMPTS, user_name, name, CONTENT3, CONTENT4, session, engine, scope_id, secret, requests1_psy)


async def send_file_to_chat(chat_id, file_url, file_name):
    await send_file_to_chat_psy(chat_id, bot, file_url, file_name)


# Все действия с ботом тг(bot.send_message) для смены этапов только здесь!!!


async def checking_for_new_message():
    await checking_for_new_message_psy(BotActivity, bot, CHAT_FOR_LOGS, history_db_psy, CONTENT1, logging, download_file_psy, session, engine, client_id, client_secret, redirect_url, domain_name, scope_id, secret, requests1_psy)


@app.post("/incomingleajson1222233js")
async def redirect_leads_to_psy(request: Request):
    await redirect_leads_psy(request, access_token_psy, STAGE_IN_AMO_1, URL_ENTITY_BASE, logging, domain_name)


@app.post("/json1222233jsdflfjblsa12")
async def handle_amo_stage_change(request: Request):
    await handle_amo_stage_change_psy(request, access_token_psy, databases_psy, URL_ENTITY_BASE, URL_USER_ID_BASE, BotActivity, STAGE_IN_AMO_1, session, logging)


@app.post('/input_handler/{text}')
async def handle_amo_message(text: str, data: IncomingMessage, request: Request):
    await handle_amo_message_psy(request, data, BotActivity, logging, session)


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=80)


async def run_bot():
    await dp.start_polling()


async def checking_messages():
    await checking_for_new_message()


# async def check_recent_messages():
#     await checking_recent_messages()


async def amo_token_update():
    while True:
        await asyncio.sleep(15 * 3600)
        access_token_psy.update_token(client_id, client_secret, redirect_url, domain_name)


async def shutdown(dispatcher: Dispatcher):
    await dispatcher.storage.close()
    await dispatcher.storage.wait_closed()


async def skip_updates():
    updates = await bot.get_updates()
    if updates:
        largest_update_id = max(update.update_id for update in updates)
        await bot.get_updates(offset=largest_update_id + 1)


async def main():
    await skip_updates()
    loop = asyncio.get_event_loop()
    fastapi_task = loop.run_in_executor(None, run_server)
    bot_task = loop.create_task(run_bot())
    messages_task = loop.create_task(checking_messages())
    # recent_messages_task = loop.create_task(check_recent_messages())
    amo_token_update_task = loop.create_task(amo_token_update())

    try:
        await asyncio.gather(bot_task, messages_task, fastapi_task, amo_token_update_task)
    except KeyboardInterrupt:
        logging.info('Stopping the application...')
        for task in asyncio.all_tasks():
            task.cancel()
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
    finally:
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        loop.close()

if __name__ == "__main__":
    asyncio.run(main())
