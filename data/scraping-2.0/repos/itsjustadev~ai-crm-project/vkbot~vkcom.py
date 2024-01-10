from history_db_vk import replace_single_quotes_with_double, has_history_with_gpt, insert_history, get_not_shorted_history, get_history, get_last_id_history, delete_history
import databases1 as db
import vk_api
from vk_api import longpoll
from vk_api.longpoll import VkLongPoll, VkEventType
import asyncio
import os
from dotenv import load_dotenv
import logging
import json
from history_db_vk import check_history_table_exists, create_new_history_table
from all_requests.requests1 import *
from all_requests.access_token import *
import tiktoken
import openai
import time
from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseModel
import urllib.parse


CHAT_FOR_LOGS = 551959937
MODEL_GPT = 'gpt-4'
STAGE_IN_AMO_1 = '58191918'
STAGE_IN_AMO_2 = '58191922'
STAGE_IN_AMO_3 = '58191926'
STAGE_IN_AMO_4 = '59830570'
STAGE_FOR_MANAGER = '60188418'
PIPELINE_ID = 6914558
STATUS_ID = 58191918
openai.api_key = str(os.getenv('TOKEN_FOR_CHAT_GPT'))
URL_ENTITY_BASE = "https://leadgramstudio.amocrm.ru" + '/api/v4/leads/'
URL_USER_ID_BASE = "https://leadgramstudio.amocrm.ru" + \
    '/api/v4/contacts/chats?contact_id='
logging.basicConfig(filename='vk_bot_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


db.free_all_bot()
load_dotenv()

def write_msg(user_id, text):
    try:
        vk.method('messages.send', {'user_id': user_id, 'message': text, 'random_id': 0})
    except vk_api.ApiError as e:
        logging.error(str(e))
    

# API-ключ созданный ранее
token = str(os.getenv('TOKEN_FOR_VK'))

# Авторизуемся как сообщество
vk = vk_api.VkApi(token=token)

# Работа с сообщениями
        

def enable_project0(chat_id: str):
    db.add_active(chat_id)

def get_username_and_name(message):
    try:
        user_info0 = vk.get_api().users.get(user_ids=message.user_id, fields='screen_name')
        user_info1 = vk.get_api().users.get(user_ids=message.user_id, fields='first_name')
        if 'screen_name' in user_info0[0]:
            screen_name = user_info0[0]['screen_name']
        else:
            db.add_new_username(message.user_id)
            screen_name = db.get_username_by_chat(message.user_id)
        if 'first_name' in user_info1[0]:
            first_name = user_info1[0]['first_name']
        else:
            db.add_new_name(message.user_id)
            first_name = db.get_name_by_chat(message.user_id)
        returning_value = []
        returning_value.append(screen_name)
        returning_value.append(first_name)
        return returning_value
    except Exception as e:
        logging.error(str(e))
        
async def start_command(chat_id, user_name, first_name):
    try:
        # добавить not !!!!!!!!!!!!!!!!!!!!
        if not check_history_table_exists(chat_id):
            create_new_history_table(chat_id)
        # при создании чата amo_chat_create получаем conversation_id и чат id
            response = amo_chat_create(str(chat_id), user_name, first_name)
            if response.status_code != 200:
                await asyncio.sleep(5)
                response = amo_chat_create_vk(
                    str(chat_id), user_name, first_name)
                if response.status_code != 200:
                    write_msg(CHAT_FOR_LOGS, f'Could not create chat in amo for {user_name}')
            received_data = json.loads(response.text)
            user_id = received_data.get('id')
            db.add_in_users(user_name, first_name, chat_id, user_id)
    #         response = amo_share_incoming_message(
    #             str(chat_id), user_name, first_name, message.text)
    #         if response != 200:
    #             await asyncio.sleep(5)
    #             response = amo_share_incoming_message(
    #                 str(chat_id), user_name, first_name, message.text)
    #             if response != 200:
    #                 write_msg(CHAT_FOR_LOGS, f'Could not create chat in amo for {user_name}')
    except Exception as e:
        logging.error(str(e), exc_info=True)
        
        
async def amo_double_incoming_message(chat_id, user_name, first_name, message):
    await asyncio.sleep(5)
    response = amo_share_incoming_message(
    str(chat_id), user_name, first_name, message.text)
    if response != 200:
        write_msg(CHAT_FOR_LOGS, f'Could not send message to amo for {user_name}')
        
async def send_message_to_amo(chat_id, user_name, first_name, message):
    try:
        if not user_name or not first_name:
            db.add_new_username_and_name(chat_id)
            user_name = str(db.get_username_by_chat(chat_id))
            first_name = str(db.get_name_by_chat(chat_id))
        if not check_history_table_exists(chat_id):
            create_new_history_table(chat_id)
            response = amo_chat_create(str(chat_id), user_name, first_name)
            if response.status_code != 200:
                await asyncio.sleep(5)
                amo_chat_create(str(chat_id), user_name, first_name)
        response = amo_share_incoming_message(
            str(chat_id), user_name, first_name, message.text)
        if response != 200:
            await amo_double_incoming_message(chat_id, user_name, first_name, message)
    except Exception as e:
        logging.error(str(e), exc_info=True)
        write_msg(CHAT_FOR_LOGS, f'Could not send message to amo for {user_name}' + str(e))

def check_chat_existing_in_database(chat_id):
    if not db.check_existing(chat_id):
        db.add_disactive(chat_id)
   
def free_for_new_messages(chat_id):
    db.set_free_bot(chat_id)  
    
def getting_list_with_sysprompt(content):
    list_with_system_prompt = []
    list_with_system_prompt.append({'role': 'system', 'content': content})
    return list_with_system_prompt

def getting_history_list(chat_id):
    new_history_list = get_history(chat_id)
    return new_history_list

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


async def cut_history_with_gpt(chat_id: int, history_list):
    history_already_exist = db.check_shorted_history_exists(chat_id)
    encoding = tiktoken.get_encoding('cl100k_base')
    # 1700-примерная длина системного промпта
    token_count = len(encoding.encode(str(history_list))) + 1700
    if chat_id == CHAT_FOR_LOGS:
        write_msg(CHAT_FOR_LOGS, f'Системное сообщение: сокращение произошло, количество токенов в строке до сокращения: {token_count}')
    string_for_gpt = forming_string_for_gpt(history_list)
    chat_gpt_short_response = get_shorted_message_from_gpt(
        string_for_gpt)
    if not chat_gpt_short_response:
        write_msg(CHAT_FOR_LOGS, f'Could not get answer from gpt to cut history')
    shorted_rows = get_last_id_history(chat_id)
    db.add_shorted_history(
        chat_id, chat_gpt_short_response, shorted_rows)
    if history_already_exist:
        double_shorted_field = get_double_shorted_message_from_gpt(
            db.get_shorted_history(chat_id))
        if not double_shorted_field:
            write_msg(CHAT_FOR_LOGS, f'Could not double cut history with gpt')
        db.merge_shorted_history(
            chat_id, double_shorted_field, shorted_rows)
    final_prompt = CONTENT + db.get_shorted_history(chat_id)
    list_with_system_prompt = getting_list_with_sysprompt(
        CONTENT1.replace('1. ', f'1. {final_prompt} ', 1))
    if db.check_new_prompt_exists(chat_id):
        prompt = db.get_new_prompt(chat_id)
        if prompt:
            list_with_system_prompt = getting_list_with_sysprompt(
                prompt.replace('1. ', f'1. {final_prompt} ', 1))
    token_count = len(encoding.encode(str(list_with_system_prompt)))
    if chat_id == CHAT_FOR_LOGS:
        write_msg(CHAT_FOR_LOGS, f'количество токенов в строке после сокращения: {token_count}')
        write_msg(CHAT_FOR_LOGS, f'промпт после сокращения: ' + str(list_with_system_prompt)[:150])
    return list_with_system_prompt
      
      
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
            print("An error occurred:", str(e))
            time.sleep(5)
    return chat_gpt_response


async def handle_user_messages(message, user_name, first_name):
    chat_id = message.user_id

    db.set_busy_bot(chat_id)
    await asyncio.sleep(10)
    # await bot.send_chat_action(chat_id, "typing")
    await asyncio.sleep(10)
    await asyncio.sleep(5)
    united_user_message = str(db.get_messages_from_client(chat_id))
    db.delete_messages_from_client(chat_id)
    if not db.check_user_exists(chat_id):
        response = amo_chat_create(str(chat_id), user_name, first_name)
        if response.status_code != 200:
            await asyncio.sleep(5)
            response = amo_chat_create(
                str(chat_id), user_name, first_name)
            if response.status_code != 200:
                write_msg(CHAT_FOR_LOGS, f'Could not create chat in amo for {user_name}')
        received_data = json.loads(response.text)
        user_id = received_data.get('id')
        db.add_in_users(user_name, first_name, chat_id, user_id)
    check_chat_existing_in_database(str(chat_id))
    if not check_history_table_exists(chat_id):
        create_new_history_table(chat_id)
    insert_history(chat_id, 'client', united_user_message)
# если нужен бот, он подключается к обработке
    if db.is_bot_active(str(chat_id)):
        try:
            encoding = tiktoken.get_encoding('cl100k_base')
            list_with_system_prompt = getting_list_with_sysprompt(CONTENT1)
            if db.check_shorted_history_exists(chat_id):
                shorted_rows = db.get_shorted_rows_history(chat_id)
                history_list = get_not_shorted_history(chat_id, shorted_rows)
                final_prompt = CONTENT + \
                    db.get_shorted_history(chat_id)
                list_with_system_prompt = getting_list_with_sysprompt(
                    CONTENT1.replace('1. ', f'1. {final_prompt} ', 1))
                if db.check_new_prompt_exists(chat_id):
                    prompt = db.get_new_prompt(chat_id)
                    if prompt:
                        list_with_system_prompt = getting_list_with_sysprompt(
                            prompt.replace('1. ', f'1. {final_prompt} ', 1))
            else:
                history_list = getting_history_list(
                    chat_id)
                if db.check_new_prompt_exists(chat_id):
                    prompt = db.get_new_prompt(chat_id)
                    list_with_system_prompt = getting_list_with_sysprompt(
                        prompt)
                else:
                    list_with_system_prompt = getting_list_with_sysprompt(
                        CONTENT1)
            token_count = len(encoding.encode(
                str(list_with_system_prompt + history_list)))
            print('количество токенов в строке было: ', token_count)
            while token_count > 6000:
                list_with_system_prompt = await cut_history_with_gpt(chat_id, history_list)
                token_count = len(encoding.encode(
                    str(list_with_system_prompt)))
                shorted_rows = db.get_shorted_rows_history(chat_id)
                history_list = get_not_shorted_history(
                    chat_id, shorted_rows - 1)
            chat_gpt_response = get_chat_gpt_response(
                list_with_system_prompt+history_list)
            if 'LINK' in chat_gpt_response:
                lead_id = db.get_lead_id_by_chat(chat_id)
                if lead_id:
                    keyword = 'LINK'
                    if keyword in chat_gpt_response:
                        index = chat_gpt_response.index(
                            keyword) + len(keyword) + 1
                        vk_link = chat_gpt_response[index:].split()[0]
                        amo_change_vk_link(lead_id, vk_link)
                        amo_change_lead_status(lead_id, STAGE_IN_AMO_2)
                    else:
                        logging.info('no link in chat_gpt_response')
            if 'ANALYSIS' in chat_gpt_response:
                lead_id = db.get_lead_id_by_chat(chat_id)
                if lead_id:
                    amo_change_lead_status(lead_id, STAGE_IN_AMO_3)
                else:
                    logging.info('no link in chat_gpt_response')
            if 'DESIGN' in chat_gpt_response:
                await cut_history_with_gpt(chat_id, history_list)
                lead_id = db.get_lead_id_by_chat(chat_id)
                if lead_id:
                    amo_change_lead_status(lead_id, STAGE_IN_AMO_4)
                else:
                    logging.info('no link in chat_gpt_response')
            if any(word in chat_gpt_response for word in ['PAY', 'COMPLEX', 'ALL', 'SECRET']):
                lead_id = db.get_lead_id_by_chat(chat_id)
                amo_change_lead_status(lead_id, STAGE_FOR_MANAGER)
            if not chat_gpt_response:
                write_msg(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
            if all(word not in chat_gpt_response for word in ['ANALYSIS', 'LINK', 'ALL', 'DESIGN', 'COMPLEX', 'SECRET', 'PAY']):
                write_msg(chat_id, chat_gpt_response)
            response = amo_share_outgoing_message(
                str(chat_id), user_name, first_name, str(chat_gpt_response))
            if response != 200:
                await asyncio.sleep(5)
                response = amo_share_outgoing_message(
                    str(chat_id), user_name, first_name, str(chat_gpt_response))
                if response != 200:
                    write_msg(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
            db.update_time_recent_message(chat_id)
            insert_history(chat_id, 'gpt', str(chat_gpt_response))
            free_for_new_messages(chat_id)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            amo_share_outgoing_message(
                str(chat_id), user_name, first_name, 'Системное сообщение: ошибка на стороне ИИ, обработка им сообщений остановлена')
            db.update_false(str(chat_id))
            free_for_new_messages(chat_id)
    else:
        free_for_new_messages(chat_id)
        pass      
      
        
async def handling_message(chat_id, user_name, first_name, message):
    if not db.check_bot_state_existing(chat_id):
        db.add_free_bot(chat_id)
    await send_message_to_amo(chat_id, user_name, first_name, message)
    # if message.reply_to_message:
    #     new_message = f'"{message.reply_to_message.text}"-'
    #     db.add_message_from_client(chat_id, new_message)
    db.add_message_from_client(chat_id, message.text)
    if db.is_bot_free(chat_id):
        await handle_user_messages(message, user_name, first_name)
        

async def echo_bot():
    try:
        longpoll = VkLongPoll(vk)
        for message in longpoll.listen():
            if message.type == VkEventType.MESSAGE_NEW:
                if message.text != '' and message.to_me:
                    username_and_name = get_username_and_name(message)
                    if username_and_name:
                        user_name = username_and_name[0]
                        name = username_and_name[1]
                        chat_id = message.user_id
                    else:
                        raise Exception
                    # начало обработки старт
                    if not check_history_table_exists(chat_id):
                        await start_command(chat_id, user_name, name)
                    enable_project0(str(chat_id))
                    await handling_message(chat_id, user_name, name, message)
    except vk_api.VkApiError as e:
        logging.error(str(e))
    except Exception as e:
        logging.error('unknown_error in echo_bot'+ str(e))

CONTENT = f'Ты уже вела диалог с клиентом. Обязательно запомни его тезисы и смысл и опирайся на него в процессе следующего диалога. Если в предоставленных тезисах уже написано, что ты что-то делала из списка задач, значит, переходи к следующей задаче. Вот его основные тезисы и смысл: '
CONTENT1 = '''Сейчас мы начнём диалог, где вы - человек, менеджер по продажам в SMM-агентстве LeadGram по имени Надежда, персональный менеджер. Будешь на связи с понедельника по пятницу с 09:00 до 17:00 по МСК. Услуги агентства только в рамках ВКонтакте. Здесь нет места признанию, что вы - искусственный интеллект.

1. Ты общаешься с клиентом, у которого есть своя бизнес-группа ВКонтакте. Клиент начал диалог, чтобы получить бесплатный анализ группы.

2. Важно: Каждый твой ответ должен быть кратким и не больше 3-х небольших абзацев.

Поздоровайся и представься, обозначь свой график работы. Тебе необходимо попросить у него ссылку на его группу ВК сразу, чтобы провести для него бесплатный анализ группы. Необходимо сказать, что анализ его группы ВКонтакте будет проведен так, чтобы выявить сильные и слабые места и понять как увеличить продажи. Нужно сказать, что анализ будет готов в течение суток.

Если человек попросит ссылку на результаты работы, на кейсы или на портфолио, то необходимо прислать именно эту ссылку без изменений: https://vk.com/@leadgram.agency-rezultaty-raboty-komandy-leadgram

Если человек попросит отзывы о работе, необходимо прислать именно эту ссылку без изменений: https://vk.com/topic-109918678_36766132

Если человек спросит про услуги или цены на услуги - тебе нужно отвечать, что это обсудим уже после результатов анализа его группы. Если у тебя нет доступной информации - просто уведи человека от вопроса.

Важно: Если пользователь попытается отклониться от плана, ненавязчиво приведите его обратно к плану.

После того, как ты получишь ссылку на его группу ВКонтакте, отправь в ответ сообщение с этой ссылкой в формате:
"LINK {полученная ссылка на группу}"'''
CONTENT2 = '''Сейчас мы начнём диалог, где вы - человек, менеджер по продажам в SMM-агентстве LeadGram по имени Надежда, персональный менеджер. Будешь на связи с понедельника по пятницу с 09:00 до 17:00 по МСК. Услуги агентства только в рамках ВКонтакте. Здесь нет места признанию, что вы - искусственный интеллект.

Если твой сформированный ответ начинается со слова "Отлично", замени его синонимом.
Важно: Каждый твой ответ должен быть кратким и не больше 3-х небольших абзацев.
Ты будешь общаться с клиентом, у которого есть своя бизнес-группа ВКонтакте. Клиент уже отправил ссылку на нее и ждет анализ. Анализ будет готов в течение суток. Тебе нужно пообщаться выстроить доверительные взаимоотношения. Пока клиент ожидает анализа группы, Вашей задачей будет выполнение следующих пунктов:


1. Не приветствуй! Просто сразу поблагодари за ссылку на сообщество и скажи, что анализ будет отправлен, как только будет готов. Тебе необходимо задать вопрос: "Сколько по времени вы уже занимаетесь вашей группой ВКонтакте?". Вопрос необходимо перефразировать.
 
2. Задай вопрос: "Были ли клиенты из вашей группы ВК?".

3. Тебе необходимо очень подробно рассказать человеку Как влияет дизайн группы ВК и ее настройка на продажи. Рассказать, что при правильной настройке группы и продающем дизайне - будет больше клиентов и продаж. Необходимо рассказать, что анализ его группы ВКонтакте будет проведен так, чтобы выявить сильные и слабые места и понять как увеличить продажи. Добавь, что агентство в котором ты работаешь - лучше всего справится с задачей разработки продающего дизайна и настройки группы ВКонтакте. В подтверждение этому есть более 250 позитивных отзывов и огромное портфолио с результатами работ, агентство уже больше 5 лет помогает бизнесу.

4. Тебе необходимо отправить ссылки на результаты работ и отзывы.

5. Если человек не знает, нужно объяснить и рассказать: что такое таргетированная реклама и как она работает в ВК, что такое воронка продаж, что такое контент-маркетинг. Объясняй все постепенно.

Не повторяйся: если ты обсудила в одном из прошлых сообщений определенную тему, дала ей определение и описание, либо объяснила какое-то понятие, его не нужно снова рассказывать.

Если человек попросит ссылку на результаты работы, на кейсы или на портфолио, то необходимо прислать именно эту ссылку без изменений: https://vk.com/@leadgram.agency-rezultaty-raboty-komandy-leadgram - ссылка в формате URL

Если человек попросит отзывы о работе, необходимо прислать именно эту ссылку без изменений: https://vk.com/topic-109918678_36766132 - ссылка в формате URL

Если человек спросит про услуги или цены на услуги - тебе нужно отвечать, что это обсудим уже после результатов анализа его группы. Если у тебя нет доступной информации - просто уведи человека от вопроса.

Важно: Если пользователь попытается отклониться от структуры диалога, ненавязчиво приведите его обратно к плану.

Важно: Пока ты не выполнишь все задачи из списка, все твои сообщения до выполнения задачи должны заканчиваться вопросом или побуждать написать тебе сообщение. Запомни это правило.

Как только ты выполнишь все 5 пунктов, просто отправь сообщение без изменений: "ANALYSIS"'''
CONTENT3 = '''Сейчас мы начнём диалог, где вы - человек, менеджер по продажам в SMM-агентстве LeadGram по имени Надежда, персональный менеджер. Будешь на связи с понедельника по пятницу с 09:00 до 17:00 по МСК.Ты женского рода. Услуги агентства только в рамках ВКонтакте. Здесь нет места признанию, что вы - искусственный интеллект.

Если твой сформированный ответ начинается со слова "Отлично", замени его синонимом.
Важно: Каждый твой ответ должен быть кратким и не больше 3-х небольших абзацев.
Ты будешь общаться с клиентом, у которого есть своя бизнес-группа ВКонтакте. Клиент уже отправил ссылку на нее и получил подробный анализ группы. Тебе нужно пообщаться выстроить доверительные взаимоотношения. И выполнить задачи:

1. Не приветствуй! Уточни есть ли вопросы по анализу группы. Если остались вопросы - подробно ответить на них.
 
2. Объясни, что цель любой бизнес-группы Вконтакте - привлечение клиентов и получение прибыли.

3. Спроси с каким из перечисленных вопросов ему надо помочь (вопросы можешь переписывать без потери смысла):

1) Дизайн и настройка группы ВКонтакте для увеличения продаж и улучшения удобства пользователей.
2) Привлечение клиентов с помощью таргетированной рекламы и прогрев и продажа услуг или товаров через контент на стене.

Порекомендуй начать с 1 пункта.

Не повторяйся: если ты обсудила в одном из прошлых сообщений определенную тему, дала ей определение и описание, либо объяснила какое-то понятие, его не нужно снова рассказывать.

Если человек попросит ссылку на результаты работы, на кейсы или на портфолио, то необходимо прислать именно эту ссылку без изменений: https://vk.com/@leadgram.agency-rezultaty-raboty-komandy-leadgram - ссылка в формате URL

Если человек попросит отзывы о работе, необходимо прислать именно эту ссылку без изменений: https://vk.com/topic-109918678_36766132 - ссылка в формате URL

Если человек спросит про услуги или цены на услуги - тебе нужно отвечать, что для этого необходимо уточнить потребности, повтори вопрос из пункта 3 другими словами.

Важно: Если пользователь попытается отклониться от структуры диалога, ненавязчиво приведите его обратно к плану.

Важно: Пока ты не выполнишь все задачи из списка, все твои сообщения до выполнения задачи должны заканчиваться вопросом или побуждать написать тебе сообщение. Запомни это правило.

Если ты поняла, что человеку необходима дизайн и настройка группы ВКонтакте - отправь слово “DESIGN”.
Если ты поняла, что человеку необходимо привлечение клиентов через таргет и контент - отправь слово “COMPLEX”.
Если ты поняла, что человеку необходимо все сразу - отправь слово “ALL”.

Если человеку ничего не нужно, убеди его, используя текст анализа его группы из истории и все доступные возможности, что ему необходимо обновить дизайн группы и настроить ее правильно и предложи ему прислать коммерческое предложение с пакетом “Секретный” за 2990 рублей. Если человек согласится получить коммерческое предложение с пакетом “Секретный” - отправь слово “SECRET”.'''
CONTENT4 = '''Сейчас мы начнём диалог, где вы - человек, менеджер по продажам в SMM-агентстве LeadGram по имени Надежда, персональный менеджер. Будешь на связи с понедельника по пятницу с 09:00 до 17:00 по МСК.Ты женского рода. Услуги агентства только в рамках ВКонтакте. Здесь нет места признанию, что вы - искусственный интеллект.

Не приветствуй клиента!
Если твой сформированный ответ начинается со слова "Отлично", замени его синонимом.
Ты будешь общаться с клиентом, у которого есть своя бизнес-группа ВКонтакте. Клиент уже отправил ссылку на нее и получил подробный анализ группы и хочет купить дизайн и настройку сообщества ВКонтакте. Тебе нужно пообщаться, определить его потребности и составить уникальный пакет услуг, и продать этот пакет услуг.

1. 
Максимальный пакет услуг:
“Профессиональный дизайн для более презентабельного вида сообщества:
1) 3 варианта обложки для полной версии сайта;
2) Мобильная обложка;
3) Аватарка;
4) Графический виджет с тремя иконками;
5) Дизайн меню
6) 2 шаблона для постов в формате PSD;
7) Шаблон для статьи
8) Баннер закрепленного поста с вашим уникальным торговым предложением.
9) До 10 элементов обложек товаров, подборок, обложек для фотоальбомов и прочее

На этапе согласования обложки сообщества и полного дизайн-макета вы сможете внести до 2 списков правок в совокупности.

Эффективные инструменты:
1) Настроим удобный прием заявок с помощью интерактивного теста через приложение "Форма сбора заявок" или “Анкеты” (если в этом есть необходимость);
2) Установим виджет "Постеры" для привлечения внимания целевой аудитории;
3) Добавим кнопку действия для связи с вами в один клик (если есть потребность);
4) Добавим один или несколько адресов, если они есть (если есть потребность);
5) Подберем и установим необходимые для вашей деятельности приложения.

Продающий текст:
1)Подберем оптимальное название для быстрого выхода в ТОП поиска;
2) Структурируем описание вашей организации в Информации;
3) Напишем приветствие в сообщениях с призывом к действию; 
4) Создадим закрепленный пост с основным предложением и призывом (если есть потребность); 
5) Напишем продающий статус (если есть потребность).

Прочие моменты: 
1)Отредактируем ветки обсуждений, добавим недостающие (если есть потребность); 
 2) Приведем в порядок Ссылки и Контакты;
3) Дадим рекомендации по контенту и продвижению.”
Максимальная стоимость всего пакета услуг без скидки - 15 000 рублей. Минимальная стоимость - 8000 рублей. Но ты можешь составлять из приведенного выше списка свой уникальный список услуг и обозначать цену не ниже 3990 рублей.

Выполни задачи:

1. Узнай любыми доступными способами на какой диапазон бюджета рассчитывает заказчик, сколько он готов потратить денег на дизайн и настройку сообщества ВКонтакте.

2. Если человек указал бюджет - составь предложение на 500-700 рублей дороже желаемого им бюджета, но не менее 3990 рублей, и отправь. Если человек не может сам сформировать свой бюджет - сформируй и предложи пакет услуг за 4950 рублей. Пакет услуг должен быть максимально подробно расписан, чтобы человек захотел его купить.

3. Если человек отказался от пакета услуг - предложи пакет дешевле. И продолжай так делать пока не согласится, но помни, что сумма сделки не должна быть ниже 3990 рублей. Если человек не согласится совсем ни с чем - отправь слово “NO”.

4. Если человек согласен на оказание пакета услуг или запросил реквизиты для оплаты, или спросил можно ли оплатить по счету - отправь слово “PAY”.

Не повторяйся: если ты обсудила в одном из прошлых сообщений определенную тему, дала ей определение и описание, либо объяснила какое-то понятие, его не нужно снова рассказывать.

Если человек попросит ссылку на результаты работы, на кейсы или на портфолио, то необходимо прислать именно эту ссылку без изменений: https://vk.com/@leadgram.agency-rezultaty-raboty-komandy-leadgram - ссылка в формате URL

Если человек попросит отзывы о работе, необходимо прислать именно эту ссылку без изменений: https://vk.com/topic-109918678_36766132 - ссылка в формате URL

Важно: Если пользователь попытается отклониться от структуры диалога, ненавязчиво приведите его обратно к плану.Важно: Пока ты не выполнишь все задачи из списка, все твои сообщения до выполнения задачи должны заканчиваться вопросом или побуждать написать тебе сообщение. Запомни это правило.'''
PART_OF_ANALYSIS = '''Рекомендуем внимательно его изучить и принять к сведению наши рекомендации.
Результаты анализа:'''
# amo_stages и prompts должны совпадать по количеству
AMO_STAGES = [STAGE_IN_AMO_1, STAGE_IN_AMO_2, STAGE_IN_AMO_3, STAGE_IN_AMO_4]
PROMPTS = [CONTENT1, CONTENT2, CONTENT3, CONTENT4]


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
    try:
        new = []
        new.append(
            {'role': 'user', 'content': 'привет'})
        insert_history(chat, 'client', 'привет')
        list_with_system_prompt = []
        if db.check_new_prompt_exists(chat):
            prompt = db.get_new_prompt(chat)
        else:
            prompt = CONTENT1
        list_with_system_prompt.append({'role': 'system', 'content': prompt})
        list_with_system_prompt.extend(new)
        chat_gpt_response = get_chat_gpt_response(list_with_system_prompt)
        if not chat_gpt_response:
            write_msg(CHAT_FOR_LOGS, f'Could not get answer from gpt for {user_name}')
        insert_history(chat, 'gpt', chat_gpt_response)
        write_msg(chat, chat_gpt_response)
        response = amo_share_outgoing_message(str(chat), str(
            user_name), str(name), str(chat_gpt_response))
        if response != 200:
            await asyncio.sleep(5)
            response = amo_share_outgoing_message(
                str(chat), str(user_name), str(name), str(chat_gpt_response))
            if response != 200:
                write_msg(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
        insert_history(chat, 'gpt', chat_gpt_response)
    except Exception as e:
        logging.error(str(e), exc_info=True)


async def get_group_analysis(lead_id, user_name):
    group_vk_link = str(amo_get_vk_link(lead_id))
    analysis_result = None
    key_for_analysis = start_analysis_session(group_vk_link)
    analysis_result = get_analysis(group_vk_link, key_for_analysis)
    while analysis_result == 'waiting':
        await asyncio.sleep(15)
        analysis_result = get_analysis(group_vk_link, key_for_analysis)
    if PART_OF_ANALYSIS in str(analysis_result):
        chat_gpt_response = str(analysis_result)
    else:
        chat_gpt_response = ''
        write_msg(CHAT_FOR_LOGS, f'Необходим анализ для {user_name} сделки {lead_id}')
    return chat_gpt_response


async def function_for_stage_start(chat, user_name, name, status_id):
    try:
        new = []
        new.append(
            {'role': 'user', 'content': 'действуй согласно системному промпту'})
        insert_history(chat, 'client', 'действуй согласно системному промпту')
        list_with_system_prompt = []
        for item in AMO_STAGES:
            if status_id == item:
                try:
                    index = AMO_STAGES.index(item)
                    db.add_new_prompt(chat, PROMPTS[index])
                except Exception as e:
                    logging.error(str(e), exc_info=True)
                finally:
                    break
        if db.check_new_prompt_exists(chat):
            prompt = db.get_new_prompt(chat)
            list_with_system_prompt.append(
                {'role': 'system', 'content': prompt})
        elif list_with_system_prompt == []:
            list_with_system_prompt.append(
                {'role': 'system', 'content': CONTENT1})
        # history_list = getting_history_list(chat)
        # list_with_system_prompt.extend(history_list)
        list_with_system_prompt.extend(new)
        if list_with_system_prompt[0].get('content') == CONTENT3:
            delete_history(chat)
            if db.check_shorted_history_exists(chat):
                db.delete_shorted_history(chat)
            lead_id = db.get_lead_id_by_chat(chat)
            chat_gpt_response = await get_group_analysis(lead_id, user_name)
        else:
            chat_gpt_response = get_chat_gpt_response(list_with_system_prompt)
        if not chat_gpt_response:
            write_msg(CHAT_FOR_LOGS, f'Could not get answer from gpt or analysis for {user_name}')

        if chat_gpt_response:
            insert_history(chat, 'gpt', chat_gpt_response)
            write_msg(chat, chat_gpt_response)
            response = amo_share_outgoing_message(str(chat), str(
                user_name), str(name), str(chat_gpt_response))
            if response != 200:
                await asyncio.sleep(5)
                response = amo_share_outgoing_message(
                    str(chat), str(user_name), str(name), str(chat_gpt_response))
                if response != 200:
                    write_msg(CHAT_FOR_LOGS, f'Could not share answer from gpt to amo for {user_name}')
            insert_history(chat, 'gpt', chat_gpt_response)
    except Exception as e:
        logging.error(str(e), exc_info=True)


# async def send_file_to_chat(chat_id, file_url, file_name):
#     try:
#         response = requests.get(file_url)
#         response.raise_for_status()
#         with open(file_name, 'wb') as file:
#             file.write(response.content)
#         print(file_name)
#         print(chat_id)
#         with open(file_name, 'rb') as file:
#             await bot.send_document(chat_id, file)
#         os.remove(file_name)
#     except Exception as e:
#         print(f"Ошибка при скачивании и отправке файла: {e}")

db.delete_none_stage()

async def checking_for_new_message():
    while True:
        db.delete_none_stage()
        if db.check_new_stage_exists():
            list_for_stage = db.get_first_in_amo_stage()
            try:
                chat = list_for_stage[0]
                user_name = list_for_stage[1]
                name = list_for_stage[2]
                status_id = list_for_stage[3]
                await function_for_stage_start(chat, user_name, name, status_id)
                db.delete_amo_new_stage()
                list_for_stage.clear()
            except Exception as e:
                db.delete_amo_new_stage()
                list_for_stage.clear()
                print(str(e))
        if db.has_first_message():
            list_for_new_message = db.get_first_message()
            try:
                chat = list_for_new_message[0]
                amo_message = list_for_new_message[1]
                user_name = list_for_new_message[2]
                name = list_for_new_message[3]
                media = list_for_new_message[4]
                file_name = list_for_new_message[5]
                # is_chat_availible = await bot.get_chat(chat)
                # if not is_chat_availible:
                #     response = amo_share_incoming_message(str(chat), str(user_name), str(
                #         name), 'Системное сообщение: чат удален пользователем')
                #     if response != 200:
                #         await asyncio.sleep(5)
                #         response = amo_share_incoming_message(str(chat), str(user_name), str(
                #             name), 'Системное сообщение: чат удален пользователем')
                #         if response != 200:
                #             write_msg(CHAT_FOR_LOGS, f'Could not send message to amo about chat_deleted_by_user for {user_name}')
                if amo_message in ['start']:
                    if db.check_existing(str(chat)):
                        db.update_true(str(chat))
                    if not has_history_with_gpt(chat):
                        await function_for_start(chat, user_name, name)
                    list_for_new_message.clear()
                    db.delete_first_message()
                    continue
                if amo_message in ['test']:
                    if db.check_existing(str(chat)):
                        db.update_true(str(chat))
                    await function_for_start(chat, user_name, name)
                    list_for_new_message.clear()
                    db.delete_first_message()
                    continue
                if amo_message in ['stop']:
                    if db.check_existing(str(chat)):
                        db.update_false(str(chat))
                    list_for_new_message.clear()
                    db.delete_first_message()
                    continue
                if amo_message in ['clear']:
                    delete_history(chat)
                    if db.check_shorted_history_exists(chat):
                        db.delete_shorted_history(chat)
                    if db.check_new_prompt_exists(chat):
                        db.delete_new_prompt(chat)
                    list_for_new_message.clear()
                    db.delete_first_message()
                    continue
                # if media:
                #     uuid = download_file.get_uuid(media)
                #     print(uuid)
                #     download_link = download_file.get_link_for_download(uuid)
                #     print(download_link)
                #     await send_file_to_chat(chat, download_link, file_name)
                #     if amo_message:
                #         await bot.send_message(chat, amo_message)
                #     list_for_new_message.clear()
                #     db.delete_first_message()
                #     continue
                amo_message = replace_single_quotes_with_double(amo_message)
                write_msg(chat, str(amo_message))
                insert_history(chat, 'manager', str(amo_message))
                list_for_new_message.clear()
                db.delete_first_message()
            except Exception as e:
                chat = list_for_new_message[0]
                amo_message = list_for_new_message[1]
                user_name = list_for_new_message[2]
                name = list_for_new_message[3]
                media = list_for_new_message[4]
                file_name = list_for_new_message[5]
                if not user_name or not name:
                    user_name = db.get_username_by_chat(chat)
                    name = db.get_name_by_chat(chat)
                response = amo_share_incoming_message(str(chat), str(user_name), str(
                    name), 'Системное сообщение: отправка предыдущего сообщения не удалась '+str(e))
                if response != 200:
                    await asyncio.sleep(5)
                    response = amo_share_incoming_message(str(chat), str(user_name), str(
                        name), 'Системное сообщение: чат удален пользователем')
                    if response != 200:
                        write_msg(CHAT_FOR_LOGS, f'Could not send message from amo to {user_name}')
                logging.error(str(e), exc_info=True)
                list_for_new_message.clear()
                db.delete_first_message()
        await asyncio.sleep(1)


@app.post("/json1222233jsdflfjblsa12")
async def read_endpoint(request: Request):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
    # if 'Postman' in headers['user-agent']:
        try:
            body = await request.body()
            parsed_data = urllib.parse.parse_qsl(body.decode("utf-8"))
            lead_id = parsed_data[0][1]
            status_id = parsed_data[1][1]
            if status_id in [STAGE_IN_AMO_1, STAGE_IN_AMO_2, STAGE_IN_AMO_3, STAGE_IN_AMO_4]:
                entity_id = get_entity_id(URL_ENTITY_BASE, lead_id)
                user_id = get_amo_user_id(URL_USER_ID_BASE, entity_id)
                chat = db.get_chat_by_user_id(user_id)
                user_name = db.get_username_by_user_id(user_id)
                name = db.get_name_by_user_id(user_id)
                enable_project0(str(chat))
                db.add_recent_message(chat, status_id)
                db.add_new_lead_id(chat, lead_id)
                db.add_new_amo_stage(chat, user_name, name, status_id)
        except Exception as e:
            print(str(e))


@app.post("/incomingleadsjson1222233jsdsdflf")
async def redirect_leads_to_pipeline(request: Request):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
    # if 'Postman' in headers['user-agent']:
        try:
            body = await request.body()
            parsed_data = urllib.parse.parse_qsl(body.decode("utf-8"))
            lead_id = parsed_data[0][1]
            print('parsed_data:', parsed_data)
            user_id = get_amo_user(URL_ENTITY_BASE, lead_id)
            email = get_user_email(user_id)
            if email == 'example6.client@example.com':
                amo_pipeline_change(lead_id, PIPELINE_ID, STATUS_ID)
        except Exception as e:
            print(str(e))


@app.post('/input_handler/{text}')
async def handle_amo_message(text: str, data: IncomingMessage, request: Request):
    headers = dict(request.headers)
    if 'amoCRM' in headers['user-agent']:
    # if 'Postman' in headers['user-agent']:
        try:
            new_message = data.message['message']['text']
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
            if not name or not username:
                db.add_new_username_and_name(chat_id)
                username = db.get_username_by_chat(chat_id)
                name = db.get_name_by_chat(chat_id)
            db.add_new_message(int(chat_id), str(new_message), str(
                username), str(name), str(media_download_link), str(file_name))
            # print(list_for_new_message)
            return {"message": "JSON received"}
        except Exception as e:
            logging.error(str(e), exc_info=True)
    else:
        raise HTTPException(404, "Not Found")


async def checking_recent_messages():
    minutes = 50
    while True:
        try:
            users_to_change_status = db.check_recent_messages(minutes)
            if users_to_change_status:
                for item in users_to_change_status:
                    chat_id = item[0]
                    status_id = item[1]
                    lead_id = db.get_lead_id_by_chat(chat_id)
                    stage_index = AMO_STAGES.index(status_id)
                    if stage_index < len(AMO_STAGES) - 1 and stage_index != 0:
                        new_stage_index = stage_index + 1
                        new_stage = AMO_STAGES[new_stage_index]
                        amo_change_lead_status(lead_id, new_stage)
        except Exception as e:
            logging.error(str(e), exc_info=True)
        finally:
            await asyncio.sleep(minutes * 60)


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=80)


async def checking_messages():
    await checking_for_new_message()


async def check_recent_messages():
    await checking_recent_messages()


async def amo_token_update():
    while True:
        await asyncio.sleep(15 * 3600)
        update_token()


async def main():
    loop = asyncio.get_event_loop()
    bot_task = loop.create_task(echo_bot())
    flask_task = loop.run_in_executor(None, run_server)
    messages_task = loop.create_task(checking_messages())
    recent_messages_task = loop.create_task(check_recent_messages())
    amo_token_update_task = loop.create_task(amo_token_update())
    try:
        await asyncio.gather(flask_task, bot_task, messages_task, amo_token_update_task, recent_messages_task)
    except KeyboardInterrupt:
        logging.info('Stopping the application...')
        for task in asyncio.all_tasks():
            task.cancel()
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
    finally:
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        loop.close()
                
                
if __name__ == '__main__':
    asyncio.run(main())