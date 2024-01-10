import asyncio
import json
from datetime import datetime, date
import random
import openai
from aioredis import Redis
from loguru import logger

from config_reader import config
from utils.gspread_tools import gs_save_new_task

save_time_long = 60 * 60 * 2
redis = Redis(host='localhost', port=6379, db=6)
openai_key = config.openai_key.get_secret_value()


# https://dialogflow.cloud.google.com/#/editAgent/mtl-skynet-hldy/


async def save_to_redis(chat_id, msg, is_answer=False):
    data_name = f'{chat_id}:{round(datetime.now().timestamp())}'
    j = {"content": msg}
    if is_answer:
        j["role"] = "assistant"
    else:
        j["role"] = "user"
    await redis.set(data_name, json.dumps(j))
    await redis.expire(data_name, save_time_long)


async def load_from_redis(chat_id):
    keys = await redis.keys(f'{chat_id}:*')
    messages = []

    for key in keys:
        value = await redis.get(key)
        # извлекаем сообщение и добавляем его в список сообщений
        message = json.loads(value)
        # извлекаем timestamp из ключа
        _, timestamp = key.decode().split(":")
        # добавляем сообщение и соответствующий timestamp в список
        messages.append((float(timestamp), message))

    # сортируем сообщения по времени
    messages.sort(key=lambda x: x[0])

    # возвращаем только сообщения, без timestamp
    return [msg for _, msg in messages]


async def delete_last_redis(chat_id):
    keys = await redis.keys(f'{chat_id}:*')

    if not keys:  # проверяем, есть ли ключи
        return

    # Извлекаем timestamp из каждого ключа и сортируем ключи по времени
    keys.sort(key=lambda key: float(key.decode().split(":")[1]))

    # Удаляем ключ с наибольшим значением времени (т.е. последний ключ)
    await redis.delete(keys[-1])


async def talk(chat_id, msg):
    await save_to_redis(chat_id, msg)
    msg_data = await load_from_redis(chat_id)
    msg = await talk_open_ai_async(msg_data=msg_data)
    if msg:
        await save_to_redis(chat_id, msg, is_answer=True)
        return msg
    else:
        await delete_last_redis(chat_id)
        return '=( connection error, retry again )='


async def talk_open_ai_list_models():
    openai.organization = "org-Iq64OmMI81NWnwcPtn72dc7E"
    openai.api_key = openai_key
    # list models
    models = openai.Model.list()
    print(list(models))
    for raw in models['data']:
        if raw['id'].find('gpt') > -1:
            print(raw['id'])
    # print(raw)
    # gpt-3.5-turbo-0613
    # gpt-3.5-turbo-16k-0613
    # gpt-3.5-turbo-16k
    # gpt-3.5-turbo-0301
    # gpt-3.5-turbo


async def talk_open_ai_async(msg=None, msg_data=None, user_name=None, b16k=False):
    openai.organization = "org-Iq64OmMI81NWnwcPtn72dc7E"
    openai.api_key = openai_key
    # list models
    # models = openai.Model.list()

    if msg_data:
        messages = msg_data
    else:
        messages = [{"role": "user", "content": msg}]
        if user_name:
            messages[0]["name"] = user_name
    try:
        print('****', messages)
        if b16k:
            chat_completion_resp = await openai.ChatCompletion.acreate(model="gpt-4", messages=messages)
        else:
            chat_completion_resp = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=messages)
        return chat_completion_resp.choices[0].message.content
    except openai.error.APIError as e:
        logger.info(e.code)
        logger.info(e.args)
        return None


async def talk_get_comment(chat_id, article):
    messages = [{"role": "system",
                 "content": "Напиши комментарий к статье, ты сторонник либертарианства. Комментарий должен быть прикольный, дружественный, не более 300 символов. Не указывай, что это комментарий или анализируй его. Напиши комментарий сразу, без введения или заключения. Не используй кавычки в ответе. Не используй хештеги # в комментариях !"},
                {"role": "user", "content": article}]
    await save_to_redis(chat_id, article)
    msg = await talk_open_ai_async(msg_data=messages)
    if msg:
        await save_to_redis(chat_id, msg, is_answer=True)
        return msg
    else:
        await delete_last_redis(chat_id)
        return '=( connection error, retry again )='


gor = (
    ("Овен", "Телец", "Близнецы", "Рак", "Лев", "Дева", "Весы", "Скорпион", "Стрелец", "Козерог", "Водолей", "Рыбы"),
    ("Расположение звезд ", "Расположение планет ", "Марс ", "Венера ", "Луна ", "Млечный путь ", "Астральные карта ",
     "Юпитер ", "Плутон ", "Сатурн ",),
    ("говорит вам ", "советует вам ", "предлагает вам ", "предрекает вам ", "благоволит вас ", "рекомендует вам ",
     "очень рекомендует вам ", "намекает вам ", "требует от вас ",),
    ("выпить пива", "напиться в хлам", "выпить никшечко", "выпить нефильтрованного никшечко", "выпить темного никшечко",
     "выпыть хугардена", "сегодня не пить =(", "хорошо приглядывать за орешками", "выпить чего по крепче",
     "пить сегодня с хорошей закуской", "поберечь печень", "нагрузить печень", "выпить ракии", "выпить дуньки",
     "выпить лозы", "выпить каспии", "сообразить на троих",)
)

lang_dict = {}


def get_horoscope() -> list:
    if date.today() == lang_dict.get('horoscope_date'):
        return lang_dict.get('horoscope', ['Ошибка при получении =('])
    else:
        today_dic = [""]
        s3 = ""
        lang_dict['horoscope_date'] = date.today()
        horoscope = ["Гороскоп на сегодня"]
        for s in gor[0]:
            horoscope.append(f'**{s}**')
            while s3 in today_dic:
                s3 = random.choice(gor[3])
            today_dic.append(s3)

            g = (random.choice(gor[1]) + random.choice(gor[2]) + s3)
            while g in horoscope:
                g = (random.choice(gor[1]) + random.choice(gor[2]) + s3)
            horoscope.append(g)

        horoscope.append("")
        horoscope.append("Желаю всем хорошего дня! 🍺🍺🍺")
        lang_dict['horoscope'] = horoscope
        return horoscope


async def talk_check_spam(article):
    messages = [{"role": "system",
                 "content": "Вы являетесь виртуальным ассистентом, специализирующимся на выявлении спама в объявлениях. Ваша задача - проанализировать предоставленные объявления и определить, являются ли они спамом. Предоставьте свой ответ в виде процентной вероятности, что данное объявление является спамом. Ваша оценка должна быть выражена в числовом формате, например, 70.0 для 70% вероятности. Никакого текста, только 2 цифры. Верни сообщение длиной в 2 символа."},
                {"role": "user", "content": article}]
    msg = None
    while msg is None:
        msg = await talk_open_ai_async(msg_data=messages)
        if not msg:
            await asyncio.sleep(1)
        if len(msg) > 3:
            logger.info(msg)
            msg = None
    return float(msg)


async def add_task_to_google(msg):
    # https://platform.openai.com/docs/guides/gpt/function-calling
    # Step 1: send the conversation and available functions to GPT
    openai.organization = "org-Iq64OmMI81NWnwcPtn72dc7E"
    openai.api_key = openai_key

    messages = [{"role": "user", "content": msg}]
    #async def gs_save_new_task(task_name, customer, manager, executor, contract_url):
    functions = [
        {
            "name": "gs_save_new_task",
            "description": "Добавляет задачу в таблицу задач",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Описание задачи",
                    },
                    "customer": {
                        "type": "string",
                        "description": "Заказчик, может быть None",
                    },
                    "manager": {
                        "type": "string",
                        "description": "Менеджер задачи, может быть None",
                    },
                    "executor": {
                        "type": "string",
                        "description": "Исполнитель, по умолчанию пользователь который дает задачу",
                    },
                    "contract_url": {
                        "type": "string",
                        "description": "Ссылка на задачу, по умолчанию ссылка на прошлое сообщение",
                    },
                    #"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["task_name", "executor", "contract_url"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "gs_save_new_task": gs_save_new_task,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        # async def gs_save_new_task(task_name, customer, manager, executor, contract_url):
        function_response = await function_to_call(
            task_name=function_args.get("task_name"),
            customer=function_args.get("customer"),
            manager=function_args.get("manager"),
            executor=function_args.get("executor"),
            contract_url=function_args.get("contract_url"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response.choices[0].message.content


async def talk_get_summary(article):
    messages = [{"role": "system",
                 "content": "Вы являетесь виртуальным ассистентом, специализирующимся на анализе текстов. Ваша задача - проанализировать предоставленный текст чата и предоставить краткий обзор его содержания."},
                {"role": "user", "content": article}]
    msg = None
    while msg is None:
        msg = await talk_open_ai_async(msg_data=messages, b16k=True)
        if not msg:
            logger.info('msg is None')
            await asyncio.sleep(3)
    return msg


if __name__ == "__main__":
    pass
    #print(asyncio.run(add_task_to_google('Скайнет, задача. Добавь задачу , заказчик эни, ссылка ya.ru , описание "Добавить новые поля в отчет"')))
    #asyncio.run(talk_open_ai_list_models())
    #exit()

    # article  = '''привет, ищу где купить мыло '''
    # print(asyncio.run(talk_check_spam(article)))
    #print(asyncio.run(talk(0,'Расскажи сказку про колобка на 10000 знаков')))
    asyncio.run(asyncio.sleep(50))
    print(asyncio.run(talk_open_ai_async('Расскажи сказку про колобка на 10000 знаков', b16k=True)))
