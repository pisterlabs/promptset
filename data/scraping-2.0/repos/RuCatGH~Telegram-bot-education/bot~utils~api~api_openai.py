import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_TOKEN')


# Получение от нейросети совета по категории
def get_advice_chatgpt(category) -> str:
    prompt = 'Можешь написать новые знания в' \
             f' теме {category} длиной в 400 символов, который будет' \
             ' полезен мне в этой сфере, и который я скорее ' \
             'всего не знаю. Пиши только совет.'

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
