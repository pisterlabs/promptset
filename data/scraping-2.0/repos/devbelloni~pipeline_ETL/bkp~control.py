from openai import OpenAI
auth = "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"
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
)

print(completion.choices[0].message)


'''
import requests as req
import json as js
import pandas as pd
# import os
import openai as gpt

url = 'https://sdw-2023-prd.up.railway.app/users'
df = pd.read_csv('SDW2023.csv')
user_ids = df['UserId'].tolist()
'''
'''
def get_user(id):
    response = req.get(f'{url}/{id}')
    if response.status_code == 200:
        data = response.json()
        with open('dados.json', 'w') as arquivo_json:
            js.dump(data, arquivo_json, indent=4)
        return data

    else:
        print(f'Erro na solicitação à API: {response.status_code}')


def update_user(id):
    response = req.put(f'{url}/{id}', js=user)
    return True if response.status_code == 200 else False


def generate_ai_news(user):
    completion = gpt.ChatCompletion.create(
  model="gpt-3.5-turbo",
        messages = [
            {
                "role":"system", "content":"Você é um especialista em marketing bancário."
            },
            {
                "role":"user", "content":f"Crie uma mensagem para {user['name']} sobre a importância dos investimentos (máximo de 100 caracteres)"
            },
        ]
    )
    return completion.choices[0].message.strip('\"')




users = [user for id in user_ids if (user := get_user(id))is not None]


gpt.api_key = "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"


for user in users:
    news = generate_ai_news(user)
    print(news)
    user['news'].append ({
        "icon" : "https://digitalinnovationone.github.io/santander-dev-week-2023-aí/icons/credit.svg",
        "description" : news
    })

    success = update_user(user)
    print(f"User {user['name']} updated? {success}!")
'''