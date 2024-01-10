import datetime
import os

import openai
import pymongo
import pytz
from dotenv import load_dotenv

from db.db import chat_col, insert_chat_to_db

load_dotenv('../.env')

api_key = os.environ.get('OPENAI_API_KEY')


def get_chatgpt_response_api(content):
    messages_to_send = [
        {"role": "system", "content": f"You are Zenith, a physical helpful robot who replies to what people say. "
                                      f"Be as concise as possible. "
                                      f"Knowledge Cutoff: 2021, "
                                      f"Date and Time as of Now: {datetime.datetime.now(pytz.timezone('Asia/Kolkata'))}"
         }
    ]
    cursor = chat_col.find({'is_revoked': False}).sort([("time", pymongo.DESCENDING)]).limit(10)
    for doc in cursor:
        if doc['query'] and doc['reply']:
            messages_to_send.insert(1, {"role": "user", "content": doc['query']})
            messages_to_send.insert(2, {"role": "assistant", "content": doc['reply']})
    messages_to_send.append({"role": "user", "content": content})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_to_send,
        api_key=api_key
    )
    return response


def get_chatgpt_reply(content):
    response = get_chatgpt_response_api(content)
    reply = response.get('choices')[0].get('message').get('content')
    insert_chat_to_db(query=content, reply=reply)
    return reply


def revoke_chats():
    chat_col.update_many(filter={'is_revoked': False}, update={'$set': {'is_revoked': True}})
