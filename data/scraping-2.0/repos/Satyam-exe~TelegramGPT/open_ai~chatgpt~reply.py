import openai
import pymongo
from openai import OpenAI

from db.db import get_api_key
from open_ai.chatgpt.constant_messages import system_message
from db.config import messages_col


def get_response(content, user_id, full_name, username):
    api_key = get_api_key(user_id=user_id, key_type='openai')
    messages_to_send = [
        {"role": "system", "content": system_message(full_name=full_name, user_id=user_id, username=username)}
    ]
    cursor = messages_col.find({'user_id': user_id}).sort([("time", pymongo.DESCENDING)]).limit(10)
    for doc in cursor:
        if not doc['is_revoked'] and doc['message'] and doc['reply']:
            messages_to_send.insert(1, {"role": "user", "content": doc['message']})
            messages_to_send.insert(2, {"role": "assistant", "content": doc['reply']})
    messages_to_send.append({"role": "user", "content": content})
    openai_client = OpenAI(api_key=api_key)
    response = openai_client.chat.completions.create(
        messages=messages_to_send,
        model='gpt-3.5-turbo',
        user=user_id
    ).model_dump()
    return response


def get_chatgpt_reply(content, user_id, full_name, username):
    response = get_response(content, user_id, full_name, username)
    return response.get('choices')[0].get('message').get('content')