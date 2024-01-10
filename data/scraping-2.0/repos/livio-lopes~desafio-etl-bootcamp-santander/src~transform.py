from extract import users
import openai

API_KEY = input('Insira sua chave OPEN_AI_KEY: ')

users_tranformed = user

def generate_ai_news(user):
  completion = openai.ChatCompletion.create(
      model='gpt-3.5-turbo',
      messages=[
          {
              "role":"system",
              "content":"Você é um especialista em marketing bancário"
          },
          {
              'role':'user',
              'content': f"Crie uma mensagem para {user['name']} sobre a importancia de investimentos (máximo de 80 caracteres)",
          }
      ]
  )
  return completion.choices[0].message.content

for user in users_tranformed:
    news = generate_ai_news(user)
    user['news'].append({
        "icon":"https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/credit.svg",
        "description": news
    })
