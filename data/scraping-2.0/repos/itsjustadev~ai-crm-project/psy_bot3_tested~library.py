import asyncio
import logging
# from sqlalchemy import engine
# import all_requests_psy.library_requests1 as requests1_psy
import json
import os
import pathlib
import openai
import time
import tiktoken
from fastapi import FastAPI, Request, HTTPException
import urllib.parse
import requests
from pydantic import BaseModel
# from databases_psy import session
# from history_db_psy import engine, metadata


async def handle_amo_message_psy(request: Request, data, BotActivity, logging, session):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
    # if 'Postman' in headers['user-agent']:
        # if 'localhost' in headers['host']:
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
                username = BotActivity.get_username_by_chat(chat_id, session)
                name = BotActivity.get_name_by_chat(chat_id, session)
            BotActivity.add_new_message(int(chat_id), str(new_message), str(
                username), str(name), str(media_download_link), str(file_name), session)
            # print(list_for_new_message)
            return {"message": "JSON received"}
        except Exception as e:
            logging.error(str(e), exc_info=True)
    else:
        raise HTTPException(404, "Not Found")


async def handle_amo_stage_change_psy(request: Request, access_token_psy, databases_psy, URL_ENTITY_BASE, URL_USER_ID_BASE, BotActivity, STAGE_IN_AMO_1, session, logging):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
    # if 'Postman' in headers['user-agent']:
        try:
            body = await request.body()
            parsed_data = urllib.parse.parse_qsl(body.decode("utf-8"))
            lead_id = parsed_data[0][1]
            status_id = parsed_data[1][1]
            entity_id = access_token_psy.get_entity_id(
                URL_ENTITY_BASE, lead_id, logging)
            user_id = access_token_psy.get_amo_user_id(
                URL_USER_ID_BASE, entity_id, logging)
            chat = BotActivity.get_chat_by_user_id(user_id, session)
            if status_id in [STAGE_IN_AMO_1]:
                print(user_id)
                print(type(user_id))
                chat = databases_psy.get_chat_by_user_id(user_id)
                print(chat, lead_id)
                if chat and lead_id:
                    databases_psy.add_new_lead_id(chat, lead_id)
        except Exception as e:
            print(str(e))

async def redirect_leads_psy(request: Request, access_token_psy, STAGE_IN_AMO_1, URL_ENTITY_BASE, logging, domain_name):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
        try:
            body = await request.body()
            parsed_data = urllib.parse.parse_qsl(body.decode("utf-8"))
            lead_id = parsed_data[0][1]
            print('parsed_data2122122233:', parsed_data)
            user_id = access_token_psy.get_amo_user(URL_ENTITY_BASE, lead_id, logging)
            print(user_id)
            email = access_token_psy.get_user_email(user_id, logging, domain_name)
            print(email)
            if email == 'example-psy.client@example.com':
                print('okokokok')
                access_token_psy.amo_change_lead_status(
                    lead_id, STAGE_IN_AMO_1, access_token_psy, domain_name)
        except Exception as e:
            print(str(e))

def send_text_to_amo(chat_id, lead_id, text, access_token_psy, BotActivity, requests1_psy, URL_ENTITY_BASE, URL_USER_ID_BASE, session, logging, scope_id, secret):
    entity_id = access_token_psy.get_entity_id(URL_ENTITY_BASE, lead_id, logging)
    user_id = access_token_psy.get_amo_user_id(URL_USER_ID_BASE, entity_id, logging)
    user_name = BotActivity.get_username_by_user_id(user_id, session)
    name = BotActivity.get_name_by_user_id(user_id, session)
    if user_name and name:
        requests1_psy.amo_share_outgoing_message(
            str(chat_id), user_name, name, text, scope_id, secret)


async def checking_for_new_message_psy(BotActivity, bot, CHAT_FOR_LOGS, history_db_psy, CONTENT1, logging, download_file_psy, session, engine, client_id, client_secret, redirect_url, domain_name, scope_id, secret, requests1_psy):
    while True:
        BotActivity.delete_none_stage(session)
        if BotActivity.check_new_stage_exists(session):
            list_for_stage = BotActivity.get_first_in_amo_stage(session)
            try:
                chat = list_for_stage[0]
                user_name = list_for_stage[1]
                name = list_for_stage[2]
                status_id = list_for_stage[3]
                BotActivity.delete_amo_new_stage(session)
                list_for_stage.clear()
            except Exception as e:
                BotActivity.delete_amo_new_stage(session)
                list_for_stage.clear()
                print(str(e))
        if BotActivity.has_first_message(session):
            list_for_new_message = BotActivity.get_first_message(session)
            try:
                chat = list_for_new_message[0]
                amo_message = list_for_new_message[1]
                user_name = list_for_new_message[2]
                name = list_for_new_message[3]
                media = list_for_new_message[4]
                file_name = list_for_new_message[5]
                is_chat_availible = await bot.get_chat(chat)
                if not is_chat_availible:
                    response = requests1_psy.amo_share_incoming_message(str(chat), str(user_name), str(
                        name), 'Системное сообщение: чат удален пользователем', scope_id, secret)
                    if response != 200:
                        await asyncio.sleep(5)
                        response = requests1_psy.amo_share_incoming_message(str(chat), str(user_name), str(
                            name), 'Системное сообщение: чат удален пользователем', scope_id, secret)
                        if response != 200:
                            await bot.send_message(CHAT_FOR_LOGS, f'Could not send message to amo about chat_deleted_by_user for {user_name}')
                if amo_message in ['start']:
                    if BotActivity.check_existing(str(chat), session):
                        BotActivity.update_true(str(chat), session)
                    if not history_db_psy.has_history_with_gpt(chat, engine):
                        await function_for_start_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, user_name, name, CHAT_FOR_LOGS, logging, session, engine, scope_id, secret, requests1_psy)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message(session)
                    continue
                if amo_message in ['resume']:
                    if BotActivity.check_existing(str(chat), session):
                        BotActivity.update_true(str(chat), session)
                    await function_for_initializing_conversation_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, CHAT_FOR_LOGS, user_name, name, logging, session, engine, scope_id, secret, requests1_psy)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message(session)
                    continue
                if amo_message in ['test']:
                    if BotActivity.check_existing(str(chat), session):
                        BotActivity.update_true(str(chat), session)
                    await function_for_start_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, user_name, name, CHAT_FOR_LOGS, logging, session, engine, scope_id, secret, requests1_psy)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message(session)
                    continue
                if amo_message in ['stop']:
                    if BotActivity.check_existing(str(chat), session):
                        BotActivity.update_false(str(chat), session)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message(session)
                    continue
                if amo_message in ['clear']:
                    history_db_psy.delete_history(chat, engine)
                    history_db_psy.delete_history_table(chat, engine)
                    if BotActivity.check_shorted_history_exists(chat, session):
                        BotActivity.delete_shorted_history(chat, session)
                    if BotActivity.check_new_prompt_exists(chat, session):
                        BotActivity.delete_new_prompt(chat, session)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message(session)
                    continue
                if media:
                    uuid = download_file_psy.get_uuid(media)
                    print(uuid)
                    download_link = download_file_psy.get_link_for_download(uuid, client_id, client_secret, redirect_url, domain_name)
                    print(download_link)
                    await send_file_to_chat_psy(chat, bot, download_link, file_name)
                    if amo_message:
                        await bot.send_message(chat, amo_message)
                    list_for_new_message.clear()
                    BotActivity.delete_first_message(session)
                    continue
                amo_message = history_db_psy.replace_single_quotes_with_double(amo_message)
                await bot.send_message(chat, str(amo_message))
                history_db_psy.insert_history(chat, 'manager', str(amo_message), engine)
                list_for_new_message.clear()
                BotActivity.delete_first_message(session)
            except Exception as e:
                chat = list_for_new_message[0]
                amo_message = list_for_new_message[1]
                user_name = list_for_new_message[2]
                name = list_for_new_message[3]
                media = list_for_new_message[4]
                file_name = list_for_new_message[5]
                if not user_name or not name:
                    user_name = BotActivity.get_username_by_chat(chat, session)
                    name = BotActivity.get_name_by_chat(chat, session)
                response = requests1_psy.amo_share_outgoing_message(str(chat), str(user_name), str(
                    name), '❗️❗️❗️Системное сообщение: отправка предыдущего сообщения не удалась ❗️❗️❗️'+'Бот заблокирован пользователем!!!', scope_id, secret)
                if response != 200:
                    await asyncio.sleep(5)
                    response = requests1_psy.amo_share_outgoing_message(str(chat), str(user_name), str(
                        name), '❗️❗️❗️Системное сообщение: чат удален пользователем❗️❗️❗️', scope_id, secret)
                    if response != 200:
                        await bot.send_message(CHAT_FOR_LOGS, f'Could not send message from amo to {user_name}')
                logging.error(str(e), exc_info=True)
                list_for_new_message.clear()
                BotActivity.delete_first_message(session)
        await asyncio.sleep(1)

def check_chat_existing_in_database(chat_id, BotActivity, session):
    if not BotActivity.check_existing(chat_id, session):
        BotActivity.add_disactive(chat_id, session)


def enable_project0(chat_id: str, BotActivity, session):
    BotActivity.add_active(chat_id, session)

async def start_command_psy(message, bot, logging, BotActivity, history_db_psy, CHAT_FOR_LOGS, session, engine, metadata, scope_id, secret, requests1_psy):
    chat_id = message.chat.id
    BotActivity.add_disactive(str(chat_id), session)
    user_name = message.from_user.username
    first_name = message.from_user.first_name
    if not BotActivity.get_username_by_chat(chat_id, session):
        BotActivity.add_new_username_and_name(chat_id, session)
        if not first_name or not user_name:
            user_name = str(BotActivity.get_username_by_chat(chat_id, session))
            first_name = str(BotActivity.get_name_by_chat(chat_id, session))
    else:
        user_name = str(message.from_user.username)
        first_name = str(message.from_user.first_name)
    try:
        history_db_psy.create_new_history_table(chat_id, engine, metadata)
        response = requests1_psy.amo_chat_create(str(chat_id), user_name, first_name, scope_id, secret)
        if response.status_code != 200:
            await asyncio.sleep(5)
            response = requests1_psy.amo_chat_create(
                str(chat_id), user_name, first_name, scope_id, secret)
            if response.status_code != 200:
                await bot.send_message(CHAT_FOR_LOGS, f'Could not create chat in amo for {user_name}')
        received_data = json.loads(response.text)
        user_id = received_data.get('id')
        BotActivity.add_in_users(user_name, first_name, chat_id, user_id, session)
    except Exception as e:
        logging.error(str(e), exc_info=True)
        await bot.send_message(CHAT_FOR_LOGS, f'Could not complete the start command in telegram for {user_name}'+str(e))


async def upload_file_psy(message, BotActivity, bot, access_token_psy, logging, CHAT_FOR_LOGS, session, client_id, client_secret, redirect_url, domain_name, scope_id, secret, requests1_psy):
    try:
        chat_id = message.chat.id
        check_chat_existing_in_database(str(chat_id), BotActivity, session)
        user_name = message.from_user.username
        first_name = message.from_user.first_name
        if not user_name or not first_name:
            user_name = str(BotActivity.get_username_by_chat(chat_id, session))
            first_name = str(BotActivity.get_name_by_chat(chat_id, session))
        else:
            user_name = str(message.from_user.username)
            first_name = str(message.from_user.first_name)
        # Проверяем, содержит ли сообщение файл
        if message.document:
            # Получаем информацию о файле
            file_info = message.document
            file_id = file_info.file_id
            file_name = file_info.file_name
            file_size = file_info.file_size
            # Получаем путь к папке "downloads" внутри проекта
            download_dir = pathlib.Path("downloads")
            # Создаем папку "downloads", если её еще нет
            download_dir.mkdir(exist_ok=True)
            # Скачиваем файл по его file_id и сохраняем его в папку "downloads"
            await bot.download_file_by_id(file_id, str(download_dir / file_name))
            answer_json = access_token_psy.upload_file_to_crm(
                file_name, f'downloads/{file_name}', file_size, client_id, client_secret, redirect_url, domain_name)
            if answer_json:
                data = json.loads(answer_json)
                link_to_download = data["_links"].get("download").get("href")
                text = ''
                if message.text:
                    text = message.text
                if message.caption:
                    text += ' ' + message.caption
                response = requests1_psy.amo_share_incoming_file(
                    str(chat_id), user_name, first_name, link_to_download, scope_id, secret)
                if response != 200:
                    await asyncio.sleep(5)
                    response = requests1_psy.amo_share_incoming_file(
                        str(chat_id), user_name, first_name, link_to_download, scope_id, secret)
                    if response != 200:
                        await bot.send_message(CHAT_FOR_LOGS, f'Could not share file to amo for {user_name}')
            try:
                os.remove(f'downloads/{file_name}')
            except Exception as e:
                print(f"Ошибка при удалении файла '{file_name}': {str(e)}")
    except Exception as e:
        logging.error(str(e), exc_info=True)
        
        
async def upload_photo_psy(message, bot, BotActivity, access_token_psy, logging, CHAT_FOR_LOGS, session, client_id, client_secret, redirect_url, domain_name, scope_id, secret, requests1_psy):
    try:
        chat_id = message.chat.id
        check_chat_existing_in_database(str(chat_id), BotActivity, session)
        user_name = message.from_user.username
        first_name = message.from_user.first_name
        if not user_name or not first_name:
            user_name = str(BotActivity.get_username_by_chat(chat_id, session))
            first_name = str(BotActivity.get_name_by_chat(chat_id, session))
        else:
            user_name = str(message.from_user.username)
            first_name = str(message.from_user.first_name)
        # Получаем информацию о фото
        # Выбираем последнее (самое крупное) изображение
        photo_info = message.photo[-1]
        file_id = photo_info.file_id
        file_size = photo_info.file_size
        file_name = f"photo_{file_id}.jpg"
        # file_name = file_name.split('-', 1)[0]
        # Получаем путь к папке "downloads" внутри проекта
        download_dir = pathlib.Path("downloads")
        # Создаем папку "downloads", если её еще нет
        download_dir.mkdir(exist_ok=True)
        # Скачиваем файл по его file_id и сохраняем его в папку "downloads"
        await bot.download_file_by_id(file_id, str(download_dir / file_name))
        answer_json = access_token_psy.upload_file_to_crm(
            file_name, f'downloads/{file_name}', file_size, client_id, client_secret, redirect_url, domain_name)
        if answer_json:
            data = json.loads(answer_json)
            link_to_download = data["_links"].get("download").get("href")
            text = ''
            if message.text:
                text = message.text
            if message.caption:
                text += ' ' + message.caption
            response = requests1_psy.amo_share_incoming_picture(
                str(chat_id), user_name, first_name, link_to_download, scope_id, secret)
            if response != 200:
                await asyncio.sleep(5)
                response = requests1_psy.amo_share_incoming_picture(
                    str(chat_id), user_name, first_name, link_to_download, scope_id, secret)
                if response != 200:
                    await bot.send_message(CHAT_FOR_LOGS, f'Could not share picture to amo for {user_name}')
        try:
            os.remove(f'downloads/{file_name}')
        except Exception as e:
            await message.answer(f"Ошибка при удалении файла '{file_name}': {str(e)}")
    except Exception as e:
        logging.error(str(e), exc_info=True)


def get_chat_gpt_response_psy(array_with_objects):
    chat_gpt_response = ''
    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model='gpt-4',
                messages=array_with_objects,
                temperature=0,
            )
            chat_gpt_response = completion.choices[0].message.content
            break
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(5)
    return chat_gpt_response


def getting_list_with_sysprompt(content):
    list_with_system_prompt = []
    list_with_system_prompt.append({'role': 'system', 'content': content})
    return list_with_system_prompt

def get_double_shorted_message_from_gpt(text):
    chat_gpt_short_response = ''
    message_to_be_short = 'сейчас я пришлю конспект диалога, сократи его без потери смысла до 2000 слов:' + \
        str(text)
    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': message_to_be_short}]
            )
            chat_gpt_short_response = completion.choices[0].message.content
            break
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(5)
    return chat_gpt_short_response

def getting_history_list(chat_id, history_db_psy, engine):
    new_history_list = history_db_psy.get_history(chat_id, engine)
    return new_history_list


async def handle_user_messages_psy(message, bot, BotActivity, history_db_psy,access_token_psy, CHAT_FOR_LOGS, logging, CONTENT1, STAGE_IN_AMO_2, session, engine, domain_name, scope_id, secret, requests1_psy):
        CONTENT = f'Ты уже вела диалог с клиентом. Обязательно запомни его тезисы и смысл и опирайся на него в процессе следующего диалога. Если в предоставленных тезисах уже написано, что ты что-то делала из списка задач, значит, переходи к следующей задаче. Вот его основные тезисы и смысл: '
        chat_id = message.chat.id
        BotActivity.set_busy_bot(chat_id, session)
        await asyncio.sleep(10)
        await bot.send_chat_action(chat_id, "typing")
        await asyncio.sleep(10)
        await bot.send_chat_action(chat_id, "typing")
        await asyncio.sleep(5)
        united_user_message = str(BotActivity.get_messages_from_client(chat_id, session))
        BotActivity.delete_messages_from_client(chat_id, session)
        try:
            history_db_psy.insert_history(chat_id, 'client', united_user_message, engine)
        except Exception as e:
            print(str(e))
        user_name = message.from_user.username
        first_name = message.from_user.first_name
        if not user_name or not first_name:
            user_name = str(BotActivity.get_username_by_chat(chat_id, session))
            first_name = str(BotActivity.get_name_by_chat(chat_id, session))
        else:
            user_name = str(message.from_user.username)
            first_name = str(message.from_user.first_name)
    # если нужен бот, он подключается к обработке
        if BotActivity.is_bot_active(str(chat_id), session):
            try:
                encoding = tiktoken.get_encoding('cl100k_base')
                list_with_system_prompt = getting_list_with_sysprompt(CONTENT1)
                if BotActivity.check_shorted_history_exists(chat_id, session):
                    shorted_rows = BotActivity.get_shorted_rows_history(chat_id, session)
                    history_list = history_db_psy.get_not_shorted_history(chat_id, shorted_rows)
                    await bot.send_chat_action(chat_id, "typing")
                    final_prompt = CONTENT + \
                        BotActivity.get_shorted_history(chat_id, session)
                    list_with_system_prompt = getting_list_with_sysprompt(
                        CONTENT1.replace('1. ', f'1. {final_prompt} ', 1))
                    if BotActivity.check_new_prompt_exists(chat_id, session):
                        prompt = BotActivity.get_new_prompt(chat_id, session)
                        if prompt:
                            list_with_system_prompt = getting_list_with_sysprompt(
                                prompt.replace('1. ', f'1. {final_prompt} ', 1))
                else:
                    history_list = getting_history_list(
                        chat_id, history_db_psy, engine)
                    if BotActivity.check_new_prompt_exists(chat_id, session):
                        prompt = BotActivity.get_new_prompt(chat_id, session)
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
                    list_with_system_prompt = await cut_history_with_gpt_psy(chat_id, history_list, BotActivity, bot, history_db_psy, CHAT_FOR_LOGS, CONTENT1, session, engine)
                    token_count = len(encoding.encode(
                        str(list_with_system_prompt)))
                    shorted_rows = BotActivity.get_shorted_rows_history(chat_id, session)
                    history_list = history_db_psy.get_not_shorted_history(
                        chat_id, shorted_rows - 1)
                chat_gpt_response = get_chat_gpt_response_psy(
                    list_with_system_prompt+history_list)
                lead_id = BotActivity.get_lead_id_by_chat(chat_id, session)
                print('lead_id', lead_id)
                if 'PAY' in chat_gpt_response:
                    access_token = str(os.getenv('ACCESS_TOKEN'))
                    lead_id = BotActivity.get_lead_id_by_chat(chat_id, session)
                    if lead_id:
                        access_token_psy.amo_change_lead_status_for_psy(
                            lead_id, STAGE_IN_AMO_2, access_token, domain_name)
                        if BotActivity.check_existing(str(chat_id), session):
                            BotActivity.update_false(str(chat_id), session)
                if 'PAY' in message.text:
                    access_token = str(os.getenv('ACCESS_TOKEN'))
                    lead_id = BotActivity.get_lead_id_by_chat(chat_id, session)
                    if lead_id:
                        access_token_psy.amo_change_lead_status_for_psy(
                            lead_id, STAGE_IN_AMO_2, access_token, domain_name)
                        if BotActivity.check_existing(str(chat_id), session):
                            BotActivity.update_false(str(chat_id), session)
                if not chat_gpt_response:
                    await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
                if all(word not in chat_gpt_response for word in ['ANALYSIS', 'LINK', 'ALL', 'DESIGN', 'COMPLEX', 'SECRET', 'PAY', 'NO']) and BotActivity.is_bot_active(str(chat_id), session):
                    await message.answer(chat_gpt_response, reply_markup=None, disable_web_page_preview=True)
                if BotActivity.is_bot_active(str(chat_id), session):
                    response = requests1_psy.amo_share_outgoing_message(
                        str(chat_id), user_name, first_name, str(chat_gpt_response), scope_id, secret)
                    if response != 200:
                        await asyncio.sleep(5)
                        response = requests1_psy.amo_share_outgoing_message(
                            str(chat_id), user_name, first_name, str(chat_gpt_response), scope_id, secret)
                        if response != 200:
                            await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
                BotActivity.update_time_recent_message(chat_id, session)
                history_db_psy.insert_history(chat_id, 'gpt', str(chat_gpt_response), engine)
                free_for_new_messages_psy(chat_id, BotActivity, session)
            except Exception as e:
                logging.error(str(e), exc_info=True)
                requests1_psy.amo_share_outgoing_message(
                    str(chat_id), user_name, first_name, 'Системное сообщение: ошибка на стороне ИИ, обработка им сообщений остановлена', scope_id, secret)
                BotActivity.update_false(str(chat_id), session)
                free_for_new_messages_psy(chat_id, BotActivity, session)
        else:
            free_for_new_messages_psy(chat_id, BotActivity, session)
            pass
        
async def function_for_stage_start_psy(bot, BotActivity, history_db_psy,CHAT_FOR_LOGS, logging, CONTENT1, AMO_STAGES, status_id, chat, PROMPTS, user_name, name, CONTENT3, CONTENT4, session, engine, scope_id, secret, requests1_psy):
    try:
        flag_analysis = 0
        new = []
        new.append(
            {'role': 'user', 'content': 'действуй согласно системному промпту'})
        history_db_psy.insert_history(chat, 'client', 'действуй согласно системному промпту', engine)
        list_with_system_prompt = []
        for item in AMO_STAGES:
            if status_id == item:
                try:
                    index = AMO_STAGES.index(item)
                    BotActivity.add_new_prompt(chat, PROMPTS[index], session)
                except Exception as e:
                    logging.error(str(e), exc_info=True)
                finally:
                    break
        if BotActivity.check_new_prompt_exists(chat, session):
            prompt = BotActivity.get_new_prompt(chat, session)
            list_with_system_prompt.append(
                {'role': 'system', 'content': prompt})
        elif list_with_system_prompt == []:
            list_with_system_prompt.append(
                {'role': 'system', 'content': CONTENT1})
        # history_list = getting_history_list(chat)
        # list_with_system_prompt.extend(history_list)
        list_with_system_prompt.extend(new)
        if list_with_system_prompt[0].get('content') == CONTENT3:
            history_db_psy.delete_history(chat, engine)
            if BotActivity.check_shorted_history_exists(chat, session):
                BotActivity.delete_shorted_history(chat, session)
            # lead_id = BotActivity.get_lead_id_by_chat(chat, session)
            # if not BotActivity.check_has_analysis(chat, session):
            #     chat_gpt_response = await get_group_analysis(lead_id, user_name)
            chat_gpt_response = ''
            flag_analysis = 1
        elif list_with_system_prompt[0].get('content') == CONTENT4:
            chat_gpt_response = 0
        else:
            chat_gpt_response = get_chat_gpt_response_psy(list_with_system_prompt)
        if not chat_gpt_response and chat_gpt_response != 0:
            requests1_psy.amo_share_outgoing_message(
                str(chat), str(user_name), str(name), '❗️❗️❗Ошибка сервера по которому получаем анализ, необходим анализ для этой сделки❗️❗️❗️', scope_id, secret)
            await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt or analysis for {user_name}')
        if chat_gpt_response != 0:
            response = requests1_psy.amo_share_outgoing_message(str(chat), str(
                user_name), str(name), str(chat_gpt_response), scope_id, secret)
            if response != 200:
                await asyncio.sleep(5)
                response = requests1_psy.amo_share_outgoing_message(
                    str(chat), str(user_name), str(name), str(chat_gpt_response), scope_id, secret)
                if response != 200:
                    await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        if chat_gpt_response:
            history_db_psy.insert_history(chat, 'gpt', chat_gpt_response, engine)
            if 'PAY' not in chat_gpt_response:
                if flag_analysis and not BotActivity.check_has_analysis(chat, session):
                    await bot.send_message(chat, chat_gpt_response, reply_markup=None)
                    BotActivity.add_has_analysis(chat, session)
                elif not flag_analysis:
                    await bot.send_message(chat, chat_gpt_response, reply_markup=None)
            if flag_analysis:
                pass
    except Exception as e:
        logging.error(str(e), exc_info=True)

        
async def function_for_initializing_conversation_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, CHAT_FOR_LOGS, user_name, name, logging, session, engine, scope_id, secret, requests1_psy):
    try:
        new = []
        list_with_system_prompt = []
        last_message_from_client = history_db_psy.get_last_message_from_client(chat, engine)
        if last_message_from_client:
            new.append(
                {'role': 'user', 'content': last_message_from_client})
            history_db_psy.insert_history(chat, 'client', last_message_from_client, engine)
        else:
            new.append(
                {'role': 'user', 'content': 'привет'})
            history_db_psy.insert_history(chat, 'client', 'привет', engine)
        if BotActivity.check_new_prompt_exists(chat, session):
            prompt = BotActivity.get_new_prompt(chat, session)
        else:
            prompt = CONTENT1
        list_with_system_prompt.append({'role': 'system', 'content': prompt})
        list_with_system_prompt.extend(new)
        chat_gpt_response = get_chat_gpt_response_psy(list_with_system_prompt)
        if not chat_gpt_response:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
        history_db_psy.insert_history(chat, 'gpt', chat_gpt_response, engine)
        await bot.send_message(chat, chat_gpt_response, reply_markup=None)
        response = requests1_psy.amo_share_outgoing_message(str(chat), str(
            user_name), str(name), str(chat_gpt_response), scope_id, secret)
        if response != 200:
            await asyncio.sleep(5)
            response = requests1_psy.amo_share_outgoing_message(
                str(chat), str(user_name), str(name), str(chat_gpt_response), scope_id, secret)
            if response != 200:
                await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        history_db_psy.insert_history(chat, 'gpt', chat_gpt_response, engine)
    except Exception as e:
        logging.error(str(e), exc_info=True)
        
async def send_file_to_chat_psy(chat_id,bot, file_url, file_name):
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

        
async def function_for_start_psy(chat, history_db_psy, BotActivity, CONTENT1, bot, user_name, name, CHAT_FOR_LOGS, logging, session, engine, scope_id, secret, requests1_psy):
    try:
        new = []
        new.append(
            {'role': 'user', 'content': 'привет'})
        history_db_psy.insert_history(chat, 'client', 'привет', engine)
        list_with_system_prompt = []
        if BotActivity.check_new_prompt_exists(chat, session):
            prompt = BotActivity.get_new_prompt(chat, session)
        else:
            prompt = CONTENT1
        list_with_system_prompt.append({'role': 'system', 'content': prompt})
        list_with_system_prompt.extend(new)
        chat_gpt_response = get_chat_gpt_response_psy(list_with_system_prompt)
        if not chat_gpt_response:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
        history_db_psy.insert_history(chat, 'gpt', chat_gpt_response, engine)
        await bot.send_message(chat, chat_gpt_response, reply_markup=None)
        response = requests1_psy.amo_share_outgoing_message(str(chat), str(
            user_name), str(name), str(chat_gpt_response), scope_id, secret)
        if response != 200:
            await asyncio.sleep(5)
            response = requests1_psy.amo_share_outgoing_message(
                str(chat), str(user_name), str(name), str(chat_gpt_response), scope_id, secret)
            if response != 200:
                await bot.send_message(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        history_db_psy.insert_history(chat, 'gpt', chat_gpt_response, engine)
    except Exception as e:
        logging.error(str(e), exc_info=True)

async def receiver_chat_gpt_psy(message, bot, logging, requests1_psy,history_db_psy, BotActivity, CHAT_FOR_LOGS, access_token_psy, CONTENT1, STAGE_IN_AMO_2, session, engine, metadata, domain_name, scope_id, secret):
    flag_first_message = False
    chat_id = message.chat.id
    if not history_db_psy.check_history_table_exists(chat_id, engine):
        flag_first_message = True
        await start_command_psy(message, bot, logging, BotActivity, history_db_psy, CHAT_FOR_LOGS, session, engine, metadata, scope_id, secret, requests1_psy)
        await share_first_messages_with_amo_psy(message, bot, logging, requests1_psy, BotActivity, CHAT_FOR_LOGS, session, scope_id, secret)
        enable_project0(str(chat_id), BotActivity, session)
    if not BotActivity.check_bot_state_existing(chat_id, session):
        BotActivity.add_free_bot(chat_id, session)
    user_name = message.from_user.username
    first_name = message.from_user.first_name
    if not user_name or not first_name:
        user_name = str(BotActivity.get_username_by_chat(chat_id, session))
        first_name = str(BotActivity.get_name_by_chat(chat_id, session))
    else:
        user_name = str(message.from_user.username)
        first_name = str(message.from_user.first_name)
    if not flag_first_message:
        try:
            requests1_psy.amo_share_incoming_message(
                str(chat_id), user_name, first_name, message.text, scope_id, secret)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            await bot.send_message(CHAT_FOR_LOGS, f'Could not send message to amo for {user_name}' + str(e))
    if message.reply_to_message:
        new_message = f'"{message.reply_to_message.text}"-'
        BotActivity.add_message_from_client(chat_id, new_message, session)
    BotActivity.add_message_from_client(chat_id, message.text, session)
    if BotActivity.is_bot_free(chat_id, session):
        asyncio.create_task(handle_user_messages_psy(message, bot, BotActivity, history_db_psy, access_token_psy, CHAT_FOR_LOGS, logging, CONTENT1, STAGE_IN_AMO_2, session, engine, domain_name, scope_id, secret, requests1_psy))

async def share_first_messages_with_amo_psy(message, bot, logging, requests1_psy, BotActivity, CHAT_FOR_LOGS, session, scope_id, secret):
    chat_id = message.chat.id
    user_name = message.from_user.username
    first_name = message.from_user.first_name
    if not user_name or not first_name:
        user_name = str(BotActivity.get_username_by_chat(chat_id, session))
        first_name = str(BotActivity.get_name_by_chat(chat_id, session))
    else:
        user_name = str(message.from_user.username)
        first_name = str(message.from_user.first_name)
    try:
        requests1_psy.amo_share_incoming_message(
            str(chat_id), user_name, first_name, message.text, scope_id, secret)
    except Exception as e:
        logging.error(str(e), exc_info=True)
        await bot.send_message(CHAT_FOR_LOGS, f'Could not send message to amo for {user_name}' + str(e))
        

async def cut_history_with_gpt_psy(chat_id: int, history_list, BotActivity, bot, history_db_psy, CHAT_FOR_LOGS, CONTENT1, session, engine):
    CONTENT = f'Ты уже вела диалог с клиентом. Обязательно запомни его тезисы и смысл и опирайся на него в процессе следующего диалога. Если в предоставленных тезисах уже написано, что ты что-то делала из списка задач, значит, переходи к следующей задаче. Вот его основные тезисы и смысл: '
    history_already_exist = BotActivity.check_shorted_history_exists(chat_id, session)
    encoding = tiktoken.get_encoding('cl100k_base')
    # 1700-примерная длина системного промпта
    token_count = len(encoding.encode(str(history_list))) + 1700
    if chat_id in (279426954, 1752794926):
        await bot.send_message(chat_id, f'Системное сообщение: сокращение произошло, количество токенов в строке до сокращения: {token_count}')
    string_for_gpt = forming_string_for_gpt_psy(history_list)
    chat_gpt_short_response = get_shorted_message_from_gpt_psy(
        string_for_gpt)
    if not chat_gpt_short_response:
        await bot.send_message(CHAT_FOR_LOGS, f'Could not get answer from gpt to cut history')
    shorted_rows = history_db_psy.get_last_id_history(chat_id, engine)
    BotActivity.add_shorted_history(
        chat_id, chat_gpt_short_response, shorted_rows, session)
    if history_already_exist:
        double_shorted_field = get_double_shorted_message_from_gpt(
            BotActivity.get_shorted_history(chat_id, session))
        if not double_shorted_field:
            await bot.send_message(CHAT_FOR_LOGS, f'Could not double cut history with gpt')
        BotActivity.merge_shorted_history(
            chat_id, double_shorted_field, shorted_rows, session)
    final_prompt = CONTENT + BotActivity.get_shorted_history(chat_id, session)
    list_with_system_prompt = getting_list_with_sysprompt(
        CONTENT1.replace('1. ', f'1. {final_prompt} ', 1))
    if BotActivity.check_new_prompt_exists(chat_id, session):
        prompt = BotActivity.get_new_prompt(chat_id, session)
        if prompt:
            list_with_system_prompt = getting_list_with_sysprompt(
                prompt.replace('1. ', f'1. {final_prompt} ', 1))
    token_count = len(encoding.encode(str(list_with_system_prompt)))
    if chat_id in (279426954, 1752794926):
        await bot.send_message(chat_id, f'количество токенов в строке после сокращения: {token_count}')
        await bot.send_message(chat_id, f'промпт после сокращения: ' + str(list_with_system_prompt)[:150])
    return list_with_system_prompt


def get_shorted_message_from_gpt_psy(text):
    chat_gpt_short_response = ''
    message_to_be_short = 'сейчас я пришлю историю сообщений между тобой и клиентом, напиши конспект диалога кратко без потери смысла до 2000 слов:' + \
        str(text)
    for _ in range(5):
        try:
            completion = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': message_to_be_short}]
            )
            chat_gpt_short_response = completion.choices[0].message.content
            break
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(5)
    return chat_gpt_short_response


def forming_string_for_gpt_psy(history_list):
    string_to_be_short = ''
    for item in history_list:
        if item['role'] == 'user':
            string_to_be_short = string_to_be_short + \
                (f"Клиент: {item['content']} ")
        elif item['role'] == 'assistant':
            string_to_be_short = string_to_be_short + \
                (f"Ты: {item['content']} ")
    return string_to_be_short


def free_for_new_messages_psy(chat_id, BotActivity, session):
    BotActivity.set_free_bot(chat_id, session)
