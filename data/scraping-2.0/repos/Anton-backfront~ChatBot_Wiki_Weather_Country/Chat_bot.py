# -------------------------------- Чат бот на базе GPT-3 (OpenAI)------------------------------------------------
from configs import TOKEN_bot, TOKEN_openAI
import openai
import telebot
# pip install --force-reinstall -v "openai==0.27.0"

openai.api_key = TOKEN_openAI


# messages=[
#         {"role": "system", "content": "You are an experienced programmer."},
#         {"role": "user", "content": "I am studying programming"},
#         {"role": "assistant", "content": "What questions do you have about programming"}
#     ]
messages=[
        # {"role": "system", "content": "You are a professional in all areas and can answer any questions"},
        # {"role": "user", "content": "I'm interested in everything"},
        # {"role": "assistant", "content": "What questions do you have"}
    ]

def update(msg, role, content):
    msg.append({"role": role, "content": content})
    return msg


def generate_response(new_text):
    global messages
    messages = update(messages, "user", new_text)
    # Объединяем вопрос и текущий контекст в единый текст
    try:

        # Используем модель GPT-3 для генерации ответа
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
            messages=messages
        )
        # Получаем первый сгенерированный ответ из списка ответов

        response_text = response['choices'][0]['message']['content']

        # Добавляем ответ в контекст


        return response_text
    except:
        return "Что-то пошло не так"














