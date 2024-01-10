import conf
import os
import openai
from extract_data import get_user, user_ids

openai.api_key = conf.API_KEY_OPENAI

def generate_ai_news(user):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "Você é um especialista em marketing bancário."
            },
            {
                "role": "user", 
                "content": f"Crie uma mensagem para {user['name']} sobre a importância dos investimentos (máximo de 100 caracteres)"
            }
        ]
    )
    
    responseChatGPT = completion.choices[0].message.content.strip('\"')
    
    return responseChatGPT

users = [user for id in user_ids if (user := get_user(id)) is not None]

for user in users:
    news = generate_ai_news(user)
    print(news)
    user['news'].append({
        "icon": "https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/credit.svg",
        "description": news
    })
