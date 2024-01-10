import openai
from openai import OpenAI

from bot.constants import bot
from db.config import api_keys_col

GET_FILE_URL = lambda path: f"https://api.telegram.org/file/bot{bot.token}/{path}"


def is_openai_api_key_set(user_id):
    if not api_keys_col.count_documents({'user_id': user_id}):
        return False
    cursor = api_keys_col.find({'user_id': user_id})
    for doc in cursor:
        if not doc['type'] == 'openai' and doc['key'] and not doc['is_expired']:
            return False
    return True


def expire_openai_api_key(user_id):
    if is_openai_api_key_set(user_id):
        api_keys_col.update_many({"user_id": user_id, "type": "openai"}, {"$set": {"is_expired": True}})


def test_openai_api_key(key):
    try:
        openai_client = OpenAI(api_key=key)
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": 'Hello'}, ],
            max_tokens=5
        ).model_dump()
        if response.get('choices')[0].get('message').get('content'):
            return True
        return False
    except openai.AuthenticationError:
        return False
