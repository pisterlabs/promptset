# from all_requests.access_token import amo_change_lead_status, amo_change_vk_link, amo_get_vk_link, amo_pipeline_change, get_amo_user, get_amo_user_id, get_analysis, get_entity_id, get_user_email, start_analysis_session, update_token, upload_file_to_crm
# from databases_psy import add_new_lead_id as psy_add_new_lead_id, get_chat_by_user_id as psy_get_chat_by_user_id, get_username_by_chat as psy_get_username, get_name_by_chat as psy_get_name, add_new_message as psy_add_new_message
# from analysis import get_analyse
from all_requests.access_token import amo_pipeline_change, get_amo_user, get_user_email
from aiogram.types import chat
import history_db as Tables
# import all_requests.requests1 as AMO_functions
from all_requests import download_file
from dotenv import load_dotenv
import all_requests.requests1 as AMO_functions
import os
from aiogram import types
from aiogram.dispatcher import Dispatcher
import openai
import logging
import asyncio
from fastapi import Request, HTTPException
import databases as BotActivity
import requests
import json
import tiktoken
import time
import urllib.parse
import datetime
import httpx
from constants import *
import pathlib
# import all_requests.access_token as AMO_connection


logging.basicConfig(filename=LOGS_PATH, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line %(lineno)d')

logger_for_stages = logging.getLogger('logger_for_stages')
logger_for_stages.setLevel(logging.INFO)
handler1 = logging.FileHandler('logger_for_stages.log')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler1.setFormatter(formatter)
logger_for_stages.addHandler(handler1)

logger_controling_deals = logging.getLogger('logger_controling_deals')
logger_controling_deals.setLevel(logging.INFO)
handler2 = logging.FileHandler('logger_controling_deals.log')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler2.setFormatter(formatter)
logger_controling_deals.addHandler(handler2)

logger_for_stage_start = logging.getLogger('logger_for_stage_start')
logger_for_stage_start.setLevel(logging.INFO)
handler3 = logging.FileHandler('logger_for_stage_start.log')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler3.setFormatter(formatter)
logger_for_stage_start.addHandler(handler3)


class AMO:
    @staticmethod
    async def send_message_to_amo(chat_id, user_name, first_name, message, bot):
        try:
            AMO_functions.amo_share_incoming_message(
                str(chat_id), user_name, first_name, message.text)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            await bot.send_message(CHAT_FOR_LOGS, f'Could not send message to amo for {user_name}' + str(e))

    @staticmethod
    async def amo_double_incoming_message(chat_id, user_name, first_name, message, bot):
        await asyncio.sleep(5)
        response = AMO_functions.amo_share_incoming_message(
            str(chat_id), user_name, first_name, message.text)
        if response != 200:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not send message to amo for {user_name}')

    @staticmethod
    async def share_first_messages_with_amo(message, bot):
        chat_id = message.chat.id
        user_name = message.from_user.username
        first_name = message.from_user.first_name
        if not user_name or not first_name:
            user_name = str(BotActivity.get_username_by_chat(chat_id))
            first_name = str(BotActivity.get_name_by_chat(chat_id))
        else:
            user_name = str(message.from_user.username)
            first_name = str(message.from_user.first_name)
        await AMO.send_message_to_amo(chat_id, user_name, first_name, message, bot)
        response = AMO_functions.amo_share_outgoing_message(
            str(chat_id), user_name, first_name, TEXT_FOR_START)
        if response != 200:
            await asyncio.sleep(5)
            response = AMO_functions.amo_share_outgoing_message(
                str(chat_id), user_name, first_name, TEXT_FOR_START)
            if response != 200:
                logging.error(
                    f'Could not share TEXT for START for {user_name}', exc_info=True)
                await bot.send_message(CHAT_FOR_LOGS, f'Could not share TEXT for START for {user_name}')
        else:
            logger_controling_deals.info(
                f'AMO text_for_start shared with {chat_id} {user_name} {first_name}'+str(datetime.datetime.now()), exc_info=True)


class BOT:
    @staticmethod
    async def start_command(message, bot):
        chat_id = message.chat.id
        BotActivity.add_disactive(str(chat_id))
        user_name = message.from_user.username
        first_name = message.from_user.first_name
        if not BotActivity.get_username_by_chat(chat_id):
            BotActivity.add_new_username_and_name(chat_id)
            if not first_name or not user_name:
                user_name = str(BotActivity.get_username_by_chat(chat_id))
                first_name = str(BotActivity.get_name_by_chat(chat_id))
        else:
            user_name = str(message.from_user.username)
            first_name = str(message.from_user.first_name)
        try:
            is_good = Tables.create_new_history_table(chat_id)
            if is_good:
                logger_controling_deals.info(
                    f'history table created for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
            await bot.send_message(chat_id, TEXT_FOR_START)
            is_good = Tables.insert_history(chat_id, 'gpt', TEXT_FOR_START)
            logger_controling_deals.info(
                f'history inserted for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
            # при создании чата amo_chat_create получаем conversation_id и чат id
            response = AMO_functions.amo_chat_create(
                str(chat_id), user_name, first_name)
            if response.status_code != 200:
                await asyncio.sleep(5)
                response = AMO_functions.amo_chat_create(
                    str(chat_id), user_name, first_name)
                if response.status_code != 200:
                    await bot.send_message(CHAT_FOR_LOGS, f'Could not create chat in amo for {user_name}')
            else:
                logger_controling_deals.info(
                    f'AMO chat created for {chat_id} {user_name} {first_name}'+str(datetime.datetime.now()), exc_info=True)
            received_data = json.loads(response.text)
            user_id = received_data.get('id')
            is_good = BotActivity.add_in_users(
                user_name, first_name, chat_id, user_id)
            if is_good:
                logger_controling_deals.info(
                    f'User added un tables for {chat_id} {user_name} {first_name}'+str(datetime.datetime.now()), exc_info=True)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            await bot.send_message(CHAT_FOR_LOGS, f'Could not complete the start command in telegram for {user_name}'+str(e))


class Database:
    @staticmethod
    def check_chat_existing(chat_id):
        if not BotActivity.check_existing(chat_id):
            BotActivity.add_disactive(chat_id)

# обработчик для отправленных пользователем фотографий и картинок


async def handle_photo(message: types.Message, bot):
    await bot.send_message(message.chat.id, 'В данный момент я могу работать только с текстовыми сообщениями')
#     try:
#         chat_id = message.chat.id
#         Database.check_chat_existing(str(chat_id))
#         user_name = message.from_user.username
#         first_name = message.from_user.first_name
#         if not user_name or not first_name:
#             user_name = str(BotActivity.get_username_by_chat(chat_id))
#             first_name = str(BotActivity.get_name_by_chat(chat_id))
#         else:
#             user_name = str(message.from_user.username)
#             first_name = str(message.from_user.first_name)
#         # Получаем информацию о фото
#         # Выбираем последнее (самое крупное) изображение
#         photo_info = message.photo[-1]
#         file_id = photo_info.file_id
#         file_size = photo_info.file_size
#         file_name = f"photo_{file_id}.jpg"
#         # file_name = file_name.split('-', 1)[0]
#         # Получаем путь к папке "downloads" внутри проекта
#         download_dir = pathlib.Path("downloads")
#         # Создаем папку "downloads", если её еще нет
#         download_dir.mkdir(exist_ok=True)
#         # Скачиваем файл по его file_id и сохраняем его в папку "downloads"
#         await bot.download_file_by_id(file_id, str(download_dir / file_name))
#         # answer_json = AMO_connection.upload_file_to_crm(
#         #     file_name, f'downloads/{file_name}', file_size)
#         if answer_json:
#             data = json.loads(answer_json)
#             link_to_download = data["_links"].get("download").get("href")
#             text = ''
#             if message.text:
#                 text = message.text
#             if message.caption:
#                 text += ' ' + message.caption
#             response = AMO_functions.amo_share_incoming_picture(
#                 str(chat_id), user_name, first_name, link_to_download)
#             if response != 200:
#                 await asyncio.sleep(5)
#                 response = AMO_functions.amo_share_incoming_picture(
#                     str(chat_id), user_name, first_name, link_to_download)
#                 if response != 200:
#                     await bot.send_message(CHAT_FOR_LOGS, f'Could not share picture to amo for {user_name}')
#         try:
#             os.remove(f'downloads/{file_name}')
#         except Exception as e:
#             await message.answer(f"Ошибка при удалении файла '{file_name}': {str(e)}")
#     except Exception as e:
#         logging.error(str(e), exc_info=True)


# обработчик для отправленных пользователем голосовых сообщений
async def voice_handler(message: types.Message, bot):
    await bot.send_message(message.chat.id, 'В данный момент я могу работать только с текстовыми сообщениями')
#     try:
#         chat_id = message.chat.id
#         Database.check_chat_existing(str(chat_id))
#         user_name = message.from_user.username
#         first_name = message.from_user.first_name
#         if not user_name or not first_name:
#             user_name = str(BotActivity.get_username_by_chat(chat_id))
#             first_name = str(BotActivity.get_name_by_chat(chat_id))
#         else:
#             user_name = str(message.from_user.username)
#             first_name = str(message.from_user.first_name)
#         if message.voice:
#             voice_info = message.voice
#             file_id = voice_info.file_id
#             file_unique_id = voice_info.file_unique_id  # Уникальный идентификатор файла
#             file_size = voice_info.file_size
#             # Путь к папке для загрузки
#             download_dir = pathlib.Path("downloads")
#             download_dir.mkdir(exist_ok=True)
#             # Формирование имени файла
#             file_name = f"{file_unique_id}.ogg"
#             # Скачивание голосового сообщения
#             await bot.download_file_by_id(file_id, str(download_dir / file_name))
#             # Обработка файла (например, загрузка в CRM)
#             answer_json = AMO_connection.upload_file_to_crm(
#                 file_name, f'downloads/{file_name}', file_size)
#             if answer_json:
#                 data = json.loads(answer_json)
#                 link_to_download = data["_links"].get("download").get("href")
#                 text = ''
#                 if message.text:
#                     text = message.text
#                 if message.caption:
#                     text += ' ' + message.caption
#                 response = AMO_functions.amo_share_incoming_voice_message(
#                     str(chat_id), user_name, first_name, link_to_download)
#                 if response != 200:
#                     await asyncio.sleep(5)
#                     response = AMO_functions.amo_share_incoming_voice_message(
#                         str(chat_id), user_name, first_name, link_to_download)
#                     if response != 200:
#                         await bot.send_message(CHAT_FOR_LOGS, f'Could not share file to amo for {user_name}')
#             # Удаление файла после обработки
#             try:
#                 os.remove(str(download_dir / file_name))
#             except Exception as e:
#                 print(f"Ошибка при удалении файла '{file_name}': {str(e)}")
#     except Exception as e:
#         logging.error(str(e), exc_info=True)


# обработчик для отправленных пользователем файлов
async def upload_file(message: types.Message, bot):
    await bot.send_message(message.chat.id, 'В данный момент я могу работать только с текстовыми сообщениями')
#     try:
#         chat_id = message.chat.id
#         Database.check_chat_existing(str(chat_id))
#         user_name = message.from_user.username
#         first_name = message.from_user.first_name
#         if not user_name or not first_name:
#             user_name = str(BotActivity.get_username_by_chat(chat_id))
#             first_name = str(BotActivity.get_name_by_chat(chat_id))
#         else:
#             user_name = str(message.from_user.username)
#             first_name = str(message.from_user.first_name)
#         # Проверяем, содержит ли сообщение файл
#         if message.document:
#             # Получаем информацию о файле
#             file_info = message.document
#             file_id = file_info.file_id
#             file_name = file_info.file_name
#             file_size = file_info.file_size
#             # Получаем путь к папке "downloads" внутри проекта
#             download_dir = pathlib.Path("downloads")
#             # Создаем папку "downloads", если её еще нет
#             download_dir.mkdir(exist_ok=True)
#             # Скачиваем файл по его file_id и сохраняем его в папку "downloads"
#             await bot.download_file_by_id(file_id, str(download_dir / file_name))
#             answer_json = AMO_connection.upload_file_to_crm(
#                 file_name, f'downloads/{file_name}', file_size)
#             if answer_json:
#                 data = json.loads(answer_json)
#                 link_to_download = data["_links"].get("download").get("href")
#                 text = ''
#                 if message.text:
#                     text = message.text
#                 if message.caption:
#                     text += ' ' + message.caption
#                 response = AMO_functions.amo_share_incoming_file(
#                     str(chat_id), user_name, first_name, link_to_download)
#                 if response != 200:
#                     await asyncio.sleep(5)
#                     response = AMO_functions.amo_share_incoming_file(
#                         str(chat_id), user_name, first_name, link_to_download)
#                     if response != 200:
#                         await bot.send_message(CHAT_FOR_LOGS, f'Could not share file to amo for {user_name}')
#             try:
#                 os.remove(f'downloads/{file_name}')
#             except Exception as e:
#                 print(f"Ошибка при удалении файла '{file_name}': {str(e)}")
#     except Exception as e:
#         logging.error(str(e), exc_info=True)


# обработчик любых текстовых сообщений
async def receiver_chat_gpt(message, bot):
    flag_first_message = False
    chat_id = message.chat.id
    if not Tables.check_history_table_exists(chat_id):
        flag_first_message = True
        await BOT.start_command(message, bot)
        await AMO.share_first_messages_with_amo(message, bot)
    if not BotActivity.check_bot_state_existing(chat_id):
        BotActivity.add_free_bot(chat_id)
    user_name = message.from_user.username
    first_name = message.from_user.first_name
    if not user_name or not first_name:
        user_name = str(BotActivity.get_username_by_chat(chat_id))
        first_name = str(BotActivity.get_name_by_chat(chat_id))
    else:
        user_name = str(message.from_user.username)
        first_name = str(message.from_user.first_name)
    if not flag_first_message:
        await AMO.send_message_to_amo(chat_id, user_name, first_name, message, bot)
    if message.reply_to_message:
        new_message = f'"{message.reply_to_message.text}"-'
        BotActivity.add_message_from_client(chat_id, new_message)
    BotActivity.add_message_from_client(chat_id, message.text)
    if BotActivity.is_bot_free(chat_id):
        asyncio.create_task(handle_user_messages(message, bot))
    else:
        logger_controling_deals.info(
            f'Bot is not free for {chat_id} '+str(datetime.datetime.now()), exc_info=True)


# обработчик для принятия заявки в канал
async def chat_join_request_handler(chat_join_request: types.ChatJoinRequest, bot):
    try:
        chat_id = chat_join_request.chat.id
        user_id = chat_join_request.from_user.id
        username = chat_join_request.from_user.username
        first_name = chat_join_request.from_user.first_name
        url = f"https://api.telegram.org/bot{TOKEN_FOR_BOT5}/approveChatJoinRequest"
        payload = {
            'chat_id': chat_id,
            'user_id': user_id
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logger_controling_deals.info(
                f'chat_join_request approved for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
        message = MessageForStart(text='/start', chat_id=user_id, username=username,
                                  first_name=first_name, reply_to_message=None)
        await receiver_chat_gpt(message, bot)
        return response.json()
    except Exception as e:
        logging.error(str(e), exc_info=True)


async def checking_messages(bot):
    await checking_for_new_message(bot)


# async def check_recent_messages(bot):
#     await checking_recent_messages(bot)


# async def amo_token_update():
#     while True:
#         await asyncio.sleep(15 * 3600)
#         update_token()


async def shutdown(dispatcher: Dispatcher):
    await dispatcher.storage.close()
    await dispatcher.storage.wait_closed()


async def skip_updates(bot):
    updates = await bot.get_updates()
    if updates:
        largest_update_id = max(update.update_id for update in updates)
        await bot.get_updates(offset=largest_update_id + 1)


# async def checking_recent_messages(bot):
#     minutes = 50
#     while True:
#         try:
#             users_to_change_status = BotActivity.check_recent_messages(minutes)
#             if users_to_change_status:
#                 for item in users_to_change_status:
#                     chat_id = item[0]
#                     try:
#                         ALL_STAGES = []
#                         ALL_STAGES.extend(AMO_STAGES)
#                         ALL_STAGES.extend(STAGE_FOR_MANAGER)
#                         status_id = item[1]
#                         time = item[2]
#                         current_time = datetime.datetime.now()
#                         time_since_last_message = current_time - time
#                         lead_id = BotActivity.get_lead_id_by_chat(chat_id)
#                         if str(lead_id) == '24976025':
#                             BotActivity.delete_recent_message(chat_id)
#                             continue
#                         if status_id == STAGE_IN_AMO_2:
#                             stage_index = AMO_STAGES.index(status_id)
#                             if stage_index < len(AMO_STAGES) - 1:
#                                 new_stage_index = stage_index + 1
#                                 new_stage = AMO_STAGES[new_stage_index]
#                                 amo_change_lead_status(lead_id, new_stage)
#                         if status_id == STAGE_IN_AMO_4:
#                             if BotActivity.get_count_to_close(chat_id) == 6 and time_since_last_message >= datetime.timedelta(hours=24):
#                                 new_stage = STAGE_FOR_MANAGER
#                                 amo_change_lead_status(lead_id, new_stage)
#                             elif BotActivity.get_count_to_close(chat_id) == 5 and time_since_last_message >= datetime.timedelta(hours=24):
#                                 await bot.send_message(chat_id, 'Здравствуйте! Подскажите, пожалуйста, когда можем ожидать обратной связи от Вас?')
#                                 BotActivity.add_recent_message(
#                                     chat_id, STAGE_IN_AMO_4)
#                                 BotActivity.add_in_deal_to_close(chat_id, 6)
#                                 send_text_to_amo(
#                                     chat_id, lead_id, 'Здравствуйте! Подскажите, пожалуйста, когда можем ожидать обратной связи от Вас?')
#                             elif BotActivity.get_count_to_close(chat_id) not in [5, 6] and time_since_last_message >= datetime.timedelta(hours=24):
#                                 await bot.send_message(chat_id, 'Здравствуйте! Вы уже приняли решение или вам нужна дополнительная информация? Я всегда на связи!')
#                                 BotActivity.add_recent_message(
#                                     chat_id, STAGE_IN_AMO_4)
#                                 BotActivity.add_in_deal_to_close(chat_id, 5)
#                                 send_text_to_amo(
#                                     chat_id, lead_id, 'Здравствуйте! Вы уже приняли решение или вам нужна дополнительная информация? Я всегда на связи!')
#                         if status_id == STAGE_IN_AMO_1:
#                             if BotActivity.get_count_to_close(chat_id) == 2 and time_since_last_message >= datetime.timedelta(hours=24):
#                                 new_amo_stage = STAGE_FOR_CLOSED_DEALS
#                                 amo_change_lead_status(lead_id, new_amo_stage)
#                                 BotActivity.delete_recent_message(chat_id)
#                                 BotActivity.delete_count_to_close(chat_id)
#                             elif BotActivity.get_count_to_close(chat_id) == 1 and time_since_last_message >= datetime.timedelta(hours=24):
#                                 await bot.send_message(chat_id, 'Вы на связи?')
#                                 BotActivity.add_recent_message(
#                                     chat_id, STAGE_IN_AMO_1)
#                                 BotActivity.add_in_deal_to_close(chat_id, 2)
#                                 send_text_to_amo(
#                                     chat_id, lead_id, 'Вы на связи?')
#                             elif not BotActivity.get_count_to_close(chat_id) and not Tables.is_vk_com_in_messages(chat_id):
#                                 await bot.send_message(chat_id, 'У вас же есть группа ВКонтакте? Пришлете ссылку?')
#                                 BotActivity.add_recent_message(
#                                     chat_id, STAGE_IN_AMO_1)
#                                 BotActivity.add_in_deal_to_close(chat_id, 1)
#                                 send_text_to_amo(
#                                     chat_id, lead_id, 'У вас же есть группа ВКонтакте? Пришлете ссылку?')
#                         if status_id == STAGE_IN_AMO_3:
#                             if BotActivity.get_count_to_close(chat_id) == 4 and time_since_last_message >= datetime.timedelta(hours=24):
#                                 new_amo_stage = STAGE_IN_AMO_4
#                                 amo_change_lead_status(lead_id, new_amo_stage)
#                             elif BotActivity.get_count_to_close(chat_id) == 3 and time_since_last_message >= datetime.timedelta(hours=24):
#                                 await bot.send_message(chat_id, 'Мне очень важно получить обратную связь. Ответьте как будет возможность.')
#                                 BotActivity.add_recent_message(
#                                     chat_id, STAGE_IN_AMO_3)
#                                 BotActivity.add_in_deal_to_close(chat_id, 4)
#                                 send_text_to_amo(
#                                     chat_id, lead_id, 'Мне очень важно получить обратную связь. Ответьте как будет возможность.')
#                             elif not BotActivity.get_count_to_close(chat_id) or BotActivity.get_count_to_close(chat_id) < 3:
#                                 if time_since_last_message >= datetime.timedelta(hours=24):
#                                     await bot.send_message(chat_id, 'У вас осталось вопросы по анализу?')
#                                     BotActivity.add_recent_message(
#                                         chat_id, STAGE_IN_AMO_3)
#                                     BotActivity.add_in_deal_to_close(
#                                         chat_id, 3)
#                                     send_text_to_amo(
#                                         chat_id, lead_id, 'У вас осталось вопросы по анализу?')
#                     except:
#                         continue
#         except Exception as e:
#             logging.error(str(e), exc_info=True)
#         finally:
#             await asyncio.sleep(30 * 60)


async def checking_for_new_message(bot):
    while True:
        BotActivity.delete_none_stage()
        if BotActivity.check_new_stage_exists():
            list_for_stage = BotActivity.get_first_in_amo_stage()
            try:
                chat = list_for_stage[0]
                user_name = list_for_stage[1]
                name = list_for_stage[2]
                status_id = list_for_stage[3]
                logger_controling_deals.info(
                    f'AMO stage change handling started for {chat} status handling {status_id} '+str(datetime.datetime.now()), exc_info=True)
                if status_id == STAGE_IN_AMO_4:
                    await bot.send_message(chat, MESSAGE_FOR_SECRET, parse_mode=types.ParseMode.MARKDOWN)
                    with open(DOCUMENT_PATH, 'rb') as document_file:
                        await bot.send_document(chat, document_file)
                    await bot.send_message(chat, TEXT_FOR_SECRET)
                    await bot.send_message(chat, TEXT_FOR_SECRET2)
                    Tables.insert_history(chat, 'gpt', MESSAGE_FOR_SECRET)
                    Tables.insert_history(
                        chat, 'gpt', TEXT_FOR_SECRET)
                    Tables.insert_history(
                        chat, 'gpt', TEXT_FOR_SECRET2)
                    AMO_functions.amo_share_outgoing_message(
                        str(chat), user_name, name, MESSAGE_FOR_SECRET)
                    AMO_functions.amo_share_outgoing_message(str(
                        chat), user_name, name, TEXT_FOR_SECRET)
                    AMO_functions.amo_share_outgoing_message(str(
                        chat), user_name, name, TEXT_FOR_SECRET2)
                if status_id in [STAGE_FOR_MANAGER]:
                    await bot.send_message(chat, CONTENT_FOR_MANAGER)
                    AMO_functions.amo_share_outgoing_message(
                        str(chat), user_name, name, CONTENT_FOR_MANAGER)
                if status_id in [STAGE_FOR_SALE]:
                    await bot.send_message(chat, CONTENT_FOR_SALE1)
                    await bot.send_message(chat, CONTENT_FOR_SALE2)
                    AMO_functions.amo_share_outgoing_message(
                        str(chat), user_name, name, CONTENT_FOR_SALE1)
                    AMO_functions.amo_share_outgoing_message(
                        str(chat), user_name, name, CONTENT_FOR_SALE2)
                if status_id not in [STAGE_IN_AMO_1, STAGE_FOR_MANAGER, STAGE_FOR_SALE, STAGE_FOR_CLOSED_DEALS, STAGE_FOR_DONE_DEALS]:
                    if status_id == STAGE_IN_AMO_3:
                        await asyncio.sleep(120)
                    logger_for_stage_start.info(
                        f'Запуск смены этапа для сделки {name}, {user_name}')
                    await function_for_stage_start(chat, user_name, name, status_id, bot)
                BotActivity.delete_amo_new_stage()
                list_for_stage.clear()
            except Exception as e:
                BotActivity.delete_amo_new_stage()
                list_for_stage.clear()
                logger_controling_deals.info(str(e), exc_info=True)
        if BotActivity.has_first_message():
            list_for_new_message = BotActivity.get_first_message()
            try:
                chat = list_for_new_message[0]
                amo_message = list_for_new_message[1]
                user_name = list_for_new_message[2]
                name = list_for_new_message[3]
                media = list_for_new_message[4]
                file_name = list_for_new_message[5]
                is_chat_availible = await bot.get_chat(chat)
                logger_controling_deals.info(
                    f'Hanling message from manager for {chat} message {amo_message} '+str(datetime.datetime.now()), exc_info=True)
                if not is_chat_availible:
                    response = AMO_functions.amo_share_incoming_message(str(chat), str(user_name), str(
                        name), 'Системное сообщение: чат удален пользователем')
                    if response != 200:
                        await asyncio.sleep(5)
                        response = AMO_functions.amo_share_incoming_message(str(chat), str(user_name), str(
                            name), 'Системное сообщение: чат удален пользователем')
                        if response != 200:
                            await bot.send_message(CHAT_FOR_LOGS, f'Could not send message to amo about chat_deleted_by_user for {user_name}')
                if amo_message in ['start']:
                    if BotActivity.check_existing(str(chat)):
                        BotActivity.update_true(str(chat))
                    if not Tables.has_history_with_gpt(chat):
                        await function_for_start(chat, user_name, name, bot)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message()
                    continue
                if amo_message in ['resume']:
                    if BotActivity.check_existing(str(chat)):
                        BotActivity.update_true(str(chat))
                    enable_project0(str(chat))
                    await function_for_initializing_conversation(chat, user_name, name, bot)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message()
                    continue
                if amo_message in ['test']:
                    if BotActivity.check_existing(str(chat)):
                        BotActivity.update_true(str(chat))
                    await function_for_start(chat, user_name, name, bot)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message()
                    continue
                if amo_message in ['stop']:
                    if BotActivity.check_existing(str(chat)):
                        BotActivity.update_false(str(chat))
                    list_for_new_message.clear()
                    BotActivity.delete_first_message()
                    continue
                if amo_message in ['clear']:
                    Tables.delete_history(chat)
                    Tables.delete_history_table(chat)
                    if BotActivity.check_shorted_history_exists(chat):
                        BotActivity.delete_shorted_history(chat)
                    if BotActivity.check_new_prompt_exists(chat):
                        BotActivity.delete_new_prompt(chat)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message()
                    continue
                if media:
                    uuid = download_file.get_uuid(media)
                    print(uuid)
                    download_link = download_file.get_link_for_download(uuid)
                    print(download_link)
                    await send_file_to_chat(chat, download_link, file_name, bot)
                    if amo_message:
                        await bot.send_message(chat, amo_message)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message()
                    continue
                amo_message = Tables.replace_single_quotes_with_double(
                    amo_message)
                await bot.send_message(chat, str(amo_message))
                Tables.insert_history(chat, 'manager', str(amo_message))
                list_for_new_message.clear()
                BotActivity.delete_first_message()
            except Exception as e:
                chat = list_for_new_message[0]
                amo_message = list_for_new_message[1]
                user_name = list_for_new_message[2]
                name = list_for_new_message[3]
                media = list_for_new_message[4]
                file_name = list_for_new_message[5]
                if not user_name or not name:
                    user_name = BotActivity.get_username_by_chat(chat)
                    name = BotActivity.get_name_by_chat(chat)
                response = AMO_functions.amo_share_outgoing_message(str(chat), str(user_name), str(
                    name), '❗️❗️❗️Системное сообщение: отправка предыдущего сообщения не удалась ❗️❗️❗️'+'Бот заблокирован пользователем!!!')
                if response != 200:
                    await asyncio.sleep(5)
                    response = AMO_functions.amo_share_outgoing_message(str(chat), str(user_name), str(
                        name), '❗️❗️❗️Системное сообщение: чат удален пользователем❗️❗️❗️')
                    if response != 200:
                        await bot.send_message(CHAT_FOR_LOGS, f'Could not send message from amo to {user_name}')
                logging.error(str(e), exc_info=True)
                list_for_new_message.clear()
                BotActivity.delete_first_message()
        await asyncio.sleep(1)


async def function_for_stage_start(chat, user_name, name, status_id, bot):
    try:
        flag_analysis = 0
        new = []
        new.append(
            {'role': 'user', 'content': 'действуй согласно системному промпту'})
        Tables.insert_history(
            chat, 'client', 'действуй согласно системному промпту')
        list_with_system_prompt = []
        for item in AMO_STAGES:
            if status_id == item:
                try:
                    index = AMO_STAGES.index(item)
                    BotActivity.add_new_prompt(chat, PROMPTS[index])
                except Exception as e:
                    logging.error(str(e), exc_info=True)
                finally:
                    break
        if BotActivity.check_new_prompt_exists(chat):
            prompt = BotActivity.get_new_prompt(chat)
            list_with_system_prompt.append(
                {'role': 'system', 'content': prompt})
        elif list_with_system_prompt == []:
            list_with_system_prompt.append(
                {'role': 'system', 'content': CONTENT1})
        list_with_system_prompt.extend(new)
        if list_with_system_prompt[0].get('content') == CONTENT3:
            # Tables.delete_history(chat)
            # if BotActivity.check_shorted_history_exists(chat):
            #     BotActivity.delete_shorted_history(chat)
            # lead_id = BotActivity.get_lead_id_by_chat(chat)
            # logger_controling_deals.info(
            #     f'Function for stage start for {chat} with system prompt {list_with_system_prompt} '+str(datetime.datetime.now()), exc_info=True)
            # # if not BotActivity.check_has_analysis(chat):
            #     # chat_gpt_response = await get_group_analysis(lead_id, user_name, bot)
            #     # if chat_gpt_response:
            #     #     logger_controling_deals.info(
            #     #         f'Analysis getted for {chat} '+str(datetime.datetime.now()), exc_info=True)
            # # else:
            chat_gpt_response = ''
            flag_analysis = 1
        elif list_with_system_prompt[0].get('content') == CONTENT4:
            chat_gpt_response = 0
        else:
            chat_gpt_response = get_chat_gpt_response(list_with_system_prompt)
        if not chat_gpt_response and chat_gpt_response != 0:
            logger_controling_deals.info(
                f'Analysis for {chat} is may not done, here it is: {chat_gpt_response} '+str(datetime.datetime.now()), exc_info=True)
            # AMO_functions.amo_share_outgoing_message(
            #     str(chat), str(user_name), str(name), '❗️❗️❗Ошибка сервера по которому получаем анализ, необходим анализ для этой сделки❗️❗️❗️')
            await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt or analysis for {user_name}')
            lead_id = BotActivity.get_lead_id_by_chat(chat)
            # amo_change_lead_status(lead_id, STAGE_FOR_MANAGER)
        # if chat_gpt_response != 0:
            # response = AMO_functions.amo_share_outgoing_message(str(chat), str(
            #     user_name), str(name), str(chat_gpt_response))
            # if response != 200:
            #     await asyncio.sleep(5)
            #     response = AMO_functions.amo_share_outgoing_message(
            #         str(chat), str(user_name), str(name), str(chat_gpt_response))
            #     if response != 200:
            #         await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        if chat_gpt_response:
            if not flag_analysis:
                Tables.insert_history(chat, 'gpt', chat_gpt_response)
            if 'PAY' not in chat_gpt_response:
                if flag_analysis and not BotActivity.check_has_analysis(chat):
                    await bot.send_message(chat, chat_gpt_response, reply_markup=None)
                    BotActivity.add_has_analysis(chat)
                elif not flag_analysis:
                    await bot.send_message(chat, chat_gpt_response, reply_markup=None)
            if flag_analysis:
                await bot.send_message(chat, TEXT_AFTER_ANALYSIS1, reply_markup=None)
                # AMO_functions.amo_share_outgoing_message(
                #     str(chat), str(user_name), str(name), TEXT_AFTER_ANALYSIS1)
                # await bot.send_message(chat, TEXT_AFTER_ANALYSIS2, reply_markup=None)
                # AMO_functions.amo_share_outgoing_message(
                #     str(chat), str(user_name), str(name), TEXT_AFTER_ANALYSIS2)
                Tables.insert_history(chat, 'gpt', TEXT_AFTER_ANALYSIS2)
    except Exception as e:
        logging.error(str(e), exc_info=True)


async def handle_user_messages(message, bot):
    chat_id = message.chat.id
    enable_project0(str(chat_id))
    BotActivity.set_busy_bot(chat_id)
    await asyncio.sleep(10)
    await bot.send_chat_action(chat_id, "typing")
    await asyncio.sleep(10)
    await bot.send_chat_action(chat_id, "typing")
    await asyncio.sleep(15)
    united_user_message = str(BotActivity.get_messages_from_client(chat_id))
    logger_controling_deals.info(
        f'Messages getted from {chat_id} {united_user_message} '+str(datetime.datetime.now()), exc_info=True)
    BotActivity.delete_messages_from_client(chat_id)
    is_good = Tables.insert_history(chat_id, 'client', united_user_message)
    if is_good:
        logger_controling_deals.info(
            f'History inserted in tables for messages from {chat_id} '+str(datetime.datetime.now()), exc_info=True)
    user_name = message.from_user.username
    first_name = message.from_user.first_name
    if not user_name or not first_name:
        user_name = str(BotActivity.get_username_by_chat(chat_id))
        first_name = str(BotActivity.get_name_by_chat(chat_id))
    else:
        user_name = str(message.from_user.username)
        first_name = str(message.from_user.first_name)
# если нужен бот, он подключается к обработке
    if BotActivity.is_bot_active(str(chat_id)):
        try:
            encoding = tiktoken.get_encoding('cl100k_base')
            list_with_system_prompt = getting_list_with_sysprompt(CONTENT1)
            if BotActivity.check_shorted_history_exists(chat_id):
                shorted_rows = BotActivity.get_shorted_rows_history(chat_id)
                history_list = Tables.get_not_shorted_history(
                    chat_id, shorted_rows)
                await bot.send_chat_action(chat_id, "typing")
                final_prompt = CONTENT + \
                    BotActivity.get_shorted_history(chat_id)
                list_with_system_prompt = getting_list_with_sysprompt(
                    CONTENT1.replace('1. ', f'1. {final_prompt} ', 1))
                if BotActivity.check_new_prompt_exists(chat_id):
                    prompt = BotActivity.get_new_prompt(chat_id)
                    if prompt:
                        list_with_system_prompt = getting_list_with_sysprompt(
                            prompt.replace('1. ', f'1. {final_prompt} ', 1))
            else:
                history_list = getting_history_list(
                    chat_id)
                if BotActivity.check_new_prompt_exists(chat_id):
                    prompt = BotActivity.get_new_prompt(chat_id)
                    list_with_system_prompt = getting_list_with_sysprompt(
                        prompt)
                else:
                    list_with_system_prompt = getting_list_with_sysprompt(
                        CONTENT1)
            token_count = len(encoding.encode(
                str(list_with_system_prompt + history_list)))
            print('количество токенов в строке было: ', token_count)
            await bot.send_chat_action(chat_id, "typing")
            while token_count > 6000:
                list_with_system_prompt = await cut_history_with_gpt(chat_id, history_list, bot)
                token_count = len(encoding.encode(
                    str(list_with_system_prompt)))
                shorted_rows = BotActivity.get_shorted_rows_history(chat_id)
                history_list = Tables.get_not_shorted_history(
                    chat_id, shorted_rows - 1)
            chat_gpt_response = get_chat_gpt_response(
                list_with_system_prompt+history_list)
            if chat_gpt_response:
                logger_controling_deals.info(
                    f'CHATGPT request done for {chat_id} system_prompt{list_with_system_prompt} '+str(datetime.datetime.now()), exc_info=True)
            if 'LINK' in chat_gpt_response:
                lead_id = BotActivity.get_lead_id_by_chat(chat_id)
                if lead_id:
                    keyword = 'LINK'
                    index = chat_gpt_response.index(
                        keyword) + len(keyword) + 1
                    vk_link = chat_gpt_response[index:].split()[0]
                    # status_code = amo_change_vk_link(lead_id, vk_link)
                    # if status_code != 200:
                    #     print(f'didnt change vklink for {lead_id} lead')
                    # else:
                    #     logger_controling_deals.info(
                    #         f'Changed vklink for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
                    # is_good = amo_change_lead_status(lead_id, STAGE_IN_AMO_2)
                    # if is_good:
                    #     logger_controling_deals.info(
                    # f'Changed status for {chat_id} to stage_in_amo2 '+str(datetime.datetime.now()), exc_info=True)
            # if 'ANALYSIS' in chat_gpt_response:
            #     lead_id = BotActivity.get_lead_id_by_chat(chat_id)
            #     if lead_id:
            #         is_good = amo_change_lead_status(lead_id, STAGE_IN_AMO_3)
            #         if is_good:
            #             logger_controling_deals.info(
            #                 f'Changed status for {chat_id} to stage_in_amo3 '+str(datetime.datetime.now()), exc_info=True)
            #     else:
            #         logging.info('no link in chat_gpt_response')
            # if 'DESIGN' in chat_gpt_response or 'SECRET' in chat_gpt_response:
            #     await cut_history_with_gpt(chat_id, history_list, bot)
            #     lead_id = BotActivity.get_lead_id_by_chat(chat_id)
            #     if lead_id:
            #         is_good = amo_change_lead_status(lead_id, STAGE_IN_AMO_4)
            #         if is_good:
            #             logger_controling_deals.info(
            #                 f'Changed status for {chat_id} to stage_in_amo4 '+str(datetime.datetime.now()), exc_info=True)
            #     else:
            #         logging.info('no link in chat_gpt_response')
            # if 'PAY' in chat_gpt_response:
            #     lead_id = BotActivity.get_lead_id_by_chat(chat_id)
            #     if lead_id:
            #         is_good = amo_change_lead_status(lead_id, STAGE_FOR_SALE)
            #         if is_good:
            #             logger_controling_deals.info(
            #                 f'Changed status for {chat_id} to stage_for_sale '+str(datetime.datetime.now()), exc_info=True)
            #         if BotActivity.check_existing(str(chat)):
            #             BotActivity.update_false(str(chat))
            #             logger_controling_deals.info(
            #                 f'bot activity turned off for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
            #     else:
            #         logging.info('no link in chat_gpt_response')
            if any(word in chat_gpt_response for word in ['COMPLEX', 'ALL', 'NO', 'PHONE']):
                lead_id = BotActivity.get_lead_id_by_chat(chat_id)
                # is_good = amo_change_lead_status(lead_id, STAGE_FOR_MANAGER)
                if is_good:
                    logger_controling_deals.info(
                        f'Changed status for {chat_id} to stage_for_manager '+str(datetime.datetime.now()), exc_info=True)
            if not chat_gpt_response:
                logger_controling_deals.info(
                    f'NO chat_gpt response for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
                await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
            if all(word not in chat_gpt_response for word in ['ANALYSIS', 'LINK', 'ALL', 'DESIGN', 'COMPLEX', 'SECRET', 'PAY', 'NO']):
                await bot.send_message(chat_id, chat_gpt_response, reply_markup=None, disable_web_page_preview=True)
            # response = AMO_functions.amo_share_outgoing_message(
            #     str(chat_id), user_name, first_name, str(chat_gpt_response))
            # if response != 200:
            #     await asyncio.sleep(5)
            #     response = AMO_functions.amo_share_outgoing_message(
            #         str(chat_id), user_name, first_name, str(chat_gpt_response))
            #     if response != 200:
            #         await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
            else:
                logger_controling_deals.info(
                    f'AMO successfully shared chat_gpt response to amo for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
            BotActivity.update_time_recent_message(chat_id)
            Tables.insert_history(chat_id, 'gpt', str(chat_gpt_response))
            free_for_new_messages(chat_id)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            # AMO_functions.amo_share_outgoing_message(
            #     str(chat_id), user_name, first_name, 'Системное сообщение: ошибка на стороне ИИ, обработка им сообщений остановлена')
            BotActivity.update_false(str(chat_id))
            free_for_new_messages(chat_id)
    else:
        logger_controling_deals.info(
            f'bot is not active for {chat_id} '+str(datetime.datetime.now()), exc_info=True)
        free_for_new_messages(chat_id)
        pass


async def helper_for_handle_amo_message(text, data: IncomingMessage, request: Request):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
        try:
            new_message = data.message['message']['text']
            media_download_link = data.message['message']['media']
            if media_download_link:
                file_name = media_download_link.rsplit('/', 1)[-1]
            else:
                file_name = ''
            client_id = data.message['receiver']['client_id']
            if 'a0c4' in client_id:
                async with httpx.AsyncClient() as client:
                    response = await client.post(f'http://localhost:80/input_handler_psy_bot/{text}', json=data.dict())
                    return {"message": "JSON received"}
            name = data.message['receiver']['name']
            conversation_id = data.message['conversation']['client_id']
            chat_id = conversation_id.split("-")[-1]
            username = client_id.split("-")[-1]
            if not name or not username or name == 'None' or username == 'None':
                username = BotActivity.get_username_by_chat(chat_id)
                name = BotActivity.get_name_by_chat(chat_id)
            is_good = BotActivity.add_new_message(int(chat_id), str(new_message), str(
                username), str(name), str(media_download_link), str(file_name))
            if is_good:
                logger_controling_deals.info(
                    f'Get new message from AMO for {chat_id} {new_message} '+str(datetime.datetime.now()), exc_info=True)
            # print(list_for_new_message)
            return {"message": "JSON received"}
        except Exception as e:
            logging.error(str(e), exc_info=True)
    else:
        raise HTTPException(404, "Not Found")


def get_double_shorted_message_from_gpt(text):
    chat_gpt_short_response = ''
    message_to_be_short = 'сейчас я пришлю конспект диалога, сократи его без потери смысла до 2000 слов:' + \
        str(text)
    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model=MODEL_GPT,
                messages=[{'role': 'user', 'content': message_to_be_short}]
            )
            chat_gpt_short_response = completion.choices[0].message.content
            break
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(5)
    return chat_gpt_short_response


async def cut_history_with_gpt(chat_id: int, history_list, bot):
    history_already_exist = BotActivity.check_shorted_history_exists(chat_id)
    encoding = tiktoken.get_encoding('cl100k_base')
    # 1700-примерная длина системного промпта
    token_count = len(encoding.encode(str(history_list))) + 1700
    if chat_id in (279426954, 1752794926):
        await bot.send_message(chat_id, f'Системное сообщение: сокращение произошло, количество токенов в строке до сокращения: {token_count}')
    string_for_gpt = forming_string_for_gpt(history_list)
    chat_gpt_short_response = get_shorted_message_from_gpt(
        string_for_gpt)
    if not chat_gpt_short_response:
        await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt to cut history')
    shorted_rows = Tables.get_last_id_history(chat_id)
    BotActivity.add_shorted_history(
        chat_id, chat_gpt_short_response, shorted_rows)
    if history_already_exist:
        double_shorted_field = get_double_shorted_message_from_gpt(
            BotActivity.get_shorted_history(chat_id))
        if not double_shorted_field:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not double cut history with gpt')
        BotActivity.merge_shorted_history(
            chat_id, double_shorted_field, shorted_rows)
    final_prompt = CONTENT + BotActivity.get_shorted_history(chat_id)
    list_with_system_prompt = getting_list_with_sysprompt(
        CONTENT1.replace('1. ', f'1. {final_prompt} ', 1))
    if BotActivity.check_new_prompt_exists(chat_id):
        prompt = BotActivity.get_new_prompt(chat_id)
        if prompt:
            list_with_system_prompt = getting_list_with_sysprompt(
                prompt.replace('1. ', f'1. {final_prompt} ', 1))
    token_count = len(encoding.encode(str(list_with_system_prompt)))
    if chat_id in (279426954, 1752794926):
        await bot.send_message(chat_id, f'количество токенов в строке после сокращения: {token_count}')
        await bot.send_message(chat_id, f'промпт после сокращения: ' + str(list_with_system_prompt)[:150])
    return list_with_system_prompt


# def send_text_to_amo(chat_id, lead_id, text):
#     entity_id = get_entity_id(URL_ENTITY_BASE, lead_id)
#     user_id = get_amo_user_id(URL_USER_ID_BASE, entity_id)
#     user_name = BotActivity.get_username_by_user_id(user_id)
#     name = BotActivity.get_name_by_user_id(user_id)
#     if user_name and name:
#         AMO_functions.amo_share_outgoing_message(
#             str(chat_id), user_name, name, text)


def forming_string_for_gpt(history_list):
    string_to_be_short = ''
    for item in history_list:
        if item['role'] == 'user':
            string_to_be_short = string_to_be_short + \
                (f"Клиент: {item['content']} ")
        elif item['role'] == 'assistant':
            string_to_be_short = string_to_be_short + \
                (f"Ты: {item['content']} ")
    return string_to_be_short


async def send_file_to_chat(chat_id, file_url, file_name, bot):
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(file_name)
        print(chat_id)
        with open(file_name, 'rb') as file:
            await bot.send_document(chat_id, file)
        os.remove(file_name)
    except Exception as e:
        print(f"Ошибка при скачивании и отправке файла: {e}")


async def function_for_initializing_conversation(chat, user_name, name, bot):
    try:
        new = []
        list_with_system_prompt = []
        last_message_from_client = Tables.get_last_message_from_client(chat)
        if last_message_from_client:
            new.append(
                {'role': 'user', 'content': last_message_from_client})
            Tables.insert_history(chat, 'client', last_message_from_client)
        else:
            new.append(
                {'role': 'user', 'content': 'привет'})
            Tables.insert_history(chat, 'client', 'привет')
        if BotActivity.check_new_prompt_exists(chat):
            prompt = BotActivity.get_new_prompt(chat)
        else:
            prompt = CONTENT1
        list_with_system_prompt.append({'role': 'system', 'content': prompt})
        list_with_system_prompt.extend(new)
        chat_gpt_response = get_chat_gpt_response(list_with_system_prompt)
        if not chat_gpt_response:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
        Tables.insert_history(chat, 'gpt', chat_gpt_response)
        await bot.send_message(chat, chat_gpt_response, reply_markup=None)
        response = AMO_functions.amo_share_outgoing_message(str(chat), str(
            user_name), str(name), str(chat_gpt_response))
        if response != 200:
            await asyncio.sleep(5)
            response = AMO_functions.amo_share_outgoing_message(
                str(chat), str(user_name), str(name), str(chat_gpt_response))
            if response != 200:
                await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        Tables.insert_history(chat, 'gpt', chat_gpt_response)
    except Exception as e:
        logging.error(str(e), exc_info=True)
        response = AMO_functions.amo_share_outgoing_message(str(chat), str(
            user_name), str(name), f'Не могу возобновить диалог по причине: {str(e)}')


# async def get_group_analysis(lead_id, user_name, bot):
#     group_vk_link = str(amo_get_vk_link(lead_id))
#     analysis_result = await get_analyse(group_vk_link)
#     if PART_OF_ANALYSIS in str(analysis_result):
#         chat_gpt_response = str(analysis_result)
#     else:
#         chat_gpt_response = ''
#         await bot.send_message(CHAT_FOR_LOGS, f'Необходим анализ для {user_name} сделки {lead_id}')
#     return chat_gpt_response


async def function_for_start(chat, user_name, name, bot):
    try:
        new = []
        new.append(
            {'role': 'user', 'content': 'привет'})
        Tables.insert_history(chat, 'client', 'привет')
        list_with_system_prompt = []
        if BotActivity.check_new_prompt_exists(chat):
            prompt = BotActivity.get_new_prompt(chat)
        else:
            prompt = CONTENT1
        list_with_system_prompt.append({'role': 'system', 'content': prompt})
        list_with_system_prompt.extend(new)
        chat_gpt_response = get_chat_gpt_response(list_with_system_prompt)
        if not chat_gpt_response:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
        Tables.insert_history(chat, 'gpt', chat_gpt_response)
        await bot.send_message(chat, chat_gpt_response, reply_markup=None)
        response = AMO_functions.amo_share_outgoing_message(str(chat), str(
            user_name), str(name), str(chat_gpt_response))
        if response != 200:
            await asyncio.sleep(5)
            response = AMO_functions.amo_share_outgoing_message(
                str(chat), str(user_name), str(name), str(chat_gpt_response))
            if response != 200:
                await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        Tables.insert_history(chat, 'gpt', chat_gpt_response)
    except Exception as e:
        logging.error(str(e), exc_info=True)


def get_shorted_message_from_gpt(text):
    chat_gpt_short_response = ''
    message_to_be_short = 'сейчас я пришлю историю сообщений между тобой и клиентом, напиши конспект диалога кратко без потери смысла до 2000 слов:' + \
        str(text)
    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model=MODEL_GPT,
                messages=[{'role': 'user', 'content': message_to_be_short}]
            )
            chat_gpt_short_response = completion.choices[0].message.content
            break
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(5)
    return chat_gpt_short_response


async def helper_for_psy_handle_amo_message(data: IncomingMessage, request: Request):
    headers = dict(request.headers)
    if 'localhost' in headers['host']:
        try:
            new_message = data.message['message']['text']
            print(new_message)
            media_download_link = data.message['message']['media']
            if media_download_link:
                file_name = media_download_link.rsplit('/', 1)[-1]
            else:
                file_name = ''
            client_id = data.message['receiver']['client_id']
            name = data.message['receiver']['name']
            conversation_id = data.message['conversation']['client_id']
            chat_id = conversation_id.split("-")[-1]
            username = client_id.split("-")[-1]
            if not name or not username or name == 'None' or username == 'None':
                username = BotActivity.get_username_by_chat(chat_id)
                name = BotActivity.get_name_by_chat(chat_id)
            is_good = BotActivity.add_new_message(int(chat_id), str(new_message), str(
                username), str(name), str(media_download_link), str(file_name))
            if is_good:
                logger_controling_deals.info(
                    f'Message from amo getted by the server for {chat_id} message {new_message} '+str(datetime.datetime.now()), exc_info=True)
            return {"message": "JSON received"}
        except Exception as e:
            logging.error(str(e), exc_info=True)
    else:
        raise HTTPException(404, "Not Found")


async def helper_for_redirect_leads(request: Request):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
        try:
            body = await request.body()
            parsed_data = urllib.parse.parse_qsl(body.decode("utf-8"))
            lead_id = parsed_data[0][1]
            print('parsed_data:', parsed_data)
            user_id = get_amo_user(URL_ENTITY_BASE, lead_id)
            email = get_user_email(user_id)
            if email == 'example6.client@example.com':
                is_good = amo_pipeline_change(lead_id, PIPELINE_ID, STATUS_ID)
                if is_good:
                    logger_controling_deals.info(
                        f'AMO pipeline changed from unsorted to test pipeline {lead_id} '+str(datetime.datetime.now()), exc_info=True)
            if email == 'example-psy.client@example.com':
                amo_pipeline_change(lead_id, 7347266, 61137218)
        except Exception as e:
            logging.error(str(e), exc_info=True)


# async def helper_for_handle_amo_stage_change(request: Request, function_for_initializing_conversation):
#     headers = dict(request.headers)
#     if 'amoCRM' in headers['user-agent']:
#         try:
#             body = await request.body()
#             parsed_data = urllib.parse.parse_qsl(body.decode("utf-8"))
#             lead_id = parsed_data[0][1]
#             status_id = parsed_data[1][1]
#             entity_id = get_entity_id(URL_ENTITY_BASE, lead_id)
#             user_id = get_amo_user_id(URL_USER_ID_BASE, entity_id)
#             logger_for_stages.info(
#                 f'Для сделки {lead_id} изменен статус на {status_id}', exc_info=True)
#             if status_id in ['61137218']:
#                 print(user_id)
#                 print(type(user_id))
#                 chat = psy_get_chat_by_user_id(user_id)
#                 print(chat, lead_id)
#                 if chat and lead_id:
#                     psy_add_new_lead_id(chat, lead_id)
#             else:
#                 chat = BotActivity.get_chat_by_user_id(user_id)
#                 logger_controling_deals.info(
#                     f'Stage changed for {chat} на {status_id} '+str(datetime.datetime.now()), exc_info=True)
#                 if status_id in [STAGE_IN_AMO_1, STAGE_IN_AMO_2, STAGE_IN_AMO_3, STAGE_IN_AMO_4]:
#                     enable_project0(str(chat))
#                     user_name = BotActivity.get_username_by_user_id(user_id)
#                     name = BotActivity.get_name_by_user_id(user_id)
#                     BotActivity.add_recent_message(chat, status_id)
#                     BotActivity.add_new_lead_id(chat, lead_id)
#                     BotActivity.add_new_amo_stage(
#                         chat, user_name, name, status_id)
#                     if status_id == STAGE_IN_AMO_1:
#                         await asyncio.sleep(150)
#                         if not Tables.get_last_message_from_gpt(chat):
#                             await function_for_initializing_conversation(chat, user_name, name)
#                 elif status_id in [STAGE_FOR_MANAGER]:
#                     BotActivity.add_recent_message(chat, status_id)
#                     if BotActivity.check_existing(str(chat)):
#                         BotActivity.update_false(str(chat))
#                 elif status_id in [STAGE_FOR_CLOSED_DEALS, STAGE_FOR_DONE_DEALS]:
#                     BotActivity.add_recent_message(chat, status_id)
#                     if BotActivity.check_existing(str(chat)):
#                         BotActivity.update_false(str(chat))
#                 elif status_id in [STAGE_FOR_SALE]:
#                     user_name = BotActivity.get_username_by_user_id(user_id)
#                     name = BotActivity.get_name_by_user_id(user_id)
#                     BotActivity.add_new_amo_stage(
#                         chat, user_name, name, status_id)
#                     BotActivity.add_recent_message(chat, status_id)
#                     if BotActivity.check_existing(str(chat)):
#                         BotActivity.update_false(str(chat))
#                 else:
#                     BotActivity.add_recent_message(chat, status_id)
#                     if BotActivity.check_existing(str(chat)):
#                         BotActivity.update_false(str(chat))
#         except Exception as e:
#             logging.error(str(e), exc_info=True)


def getting_list_with_sysprompt(content):
    list_with_system_prompt = []
    list_with_system_prompt.append({'role': 'system', 'content': content})
    return list_with_system_prompt


def getting_history_list(chat_id):
    new_history_list = Tables.get_history(chat_id)
    return new_history_list


def get_chat_gpt_response(array_with_objects):
    chat_gpt_response = ''
    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model=MODEL_GPT,
                messages=array_with_objects,
                temperature=0,
            )
            chat_gpt_response = completion.choices[0].message.content
            break
        except Exception as e:
            logging.error(str(e), exc_info=True)
            time.sleep(5)
    return chat_gpt_response


def check_string_contains_substring(substring, string):
    words = substring.split()
    i = 0
    for word in words:
        if word in string:
            i += 1
        else:
            pass
    return True if i / len(words) >= 0.6 else False


def free_for_new_messages(chat_id):
    BotActivity.set_free_bot(chat_id)


def enable_project0(chat_id: str):
    BotActivity.add_active(chat_id)
