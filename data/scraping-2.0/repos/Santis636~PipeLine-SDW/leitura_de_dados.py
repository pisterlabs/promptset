import openai
import pandas as pd
import requests
import json


SDW2023_url = 'https://sdw-2023-prd.up.railway.app'
openai_key = 'Chave-GPT'
openai.api_key = openai_key 

df = pd.read_csv('users.csv')
user_ids = df['UserID']


def get_user(id):
    response = requests.get(f'{SDW2023_url}/users/{id}')
    return response.json() if response.status_code == 200 else None

def generate_message(user):
    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        message = [
            {
            'role': 'system',
            'content': 'Você será um especialista bancario que tem conhecimentos abrangentes em marketing.'
            },
            {
                'role':'user',
                'content':f'Crie uma mensagem para {user["name"]} de como é importante investir e fazer seu dinheiro render, sendo objetiva (100 caracteres máximos)'
            }
        ]
    )
    return completion.choice[0].message.content.strip('\"')

def update_user(user):
    response = requests.put(f'{SDW2023_url}/users/{user["id"]}', json=user)
    return True if response.status_code == 200 else False

users = [user for id in user_ids 
         if (user := get_user(id)) is not None]
print(json.dumps(users, indent= 2))

for user in users:
        news = generate_message(user)
        user['news'].append({
            'icon':'https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/credit.svg',
            'description': news
        })

for user in users:
    sucess = update_user(user)
    print(f'User {user["name"]} update? {"Atualizado com sucesso" if user["news"] is not None else "falha na atualizacao"}')
print(json.dumps(users, indent= 2))
    