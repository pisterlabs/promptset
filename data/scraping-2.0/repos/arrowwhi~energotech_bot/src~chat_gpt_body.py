import time

import openai
import requests

from config import openai_api_key


MODEL = 'gpt-3.5-turbo-0301'

users = {}

openai.api_key = openai_api_key


def _get_clear_history():
    common_start = f"""Ты полезный ассистент с ИИ, который готов помочь своему пользователю. 
    Ты даешь полезные советы по поводу электротехнического оборудования. 
    Если тебя спрашивают про что-то другое - ты говоришь, что создан не для этого и не можешь отвечать на такие вопросы.
    """
    return [{"role": "system", "content": common_start}]


def _get_user(user_id):
    user_id = str(user_id)
    user = users.get(
        user_id, {'id': user_id, 'history': _get_clear_history(), 'last_prompt_time': 0})
    users[user_id] = user   
    return user


def openAI(user_id, rq):
    user_id = str(user_id)
    user = _get_user(user_id)

    # Drop history if user is inactive for 1 hour
    if time.time() - user['last_prompt_time'] > 60 * 60:
        user['last_prompt_time'] = 0
        user['history'] = _get_clear_history()

    if rq and 0 < len(rq) < 3000:
        user['history'].append({"role": "user", "content": rq})

        completion = openai.ChatCompletion.create(
            model=MODEL, messages=user['history'], temperature=0.7)
        ans = completion['choices'][0]['message']['content']

        user['history'].append({"role": "assistant", "content": ans})
        user['last_prompt_time'] = time.time()
        return ans
