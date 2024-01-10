'''
Contexto:
Você é um Cientista de Dados no Santander e recebeu a tarefa de envolver seus clientes de maneira mais personalizada. Seu objetivo é usar o poder da IA Generativa para criar mensagens de marketing personalizadas que serão entregues a cada cliente.

1. Você recebeu uma planilha simples, em formato CSV ('SDW2023.csv'), com uma lista de IDs de usuário do banco
2. Seu trabalho é consumir o endpoint GET https://sdw-2023-prd.up.raiway.app/users/{id} (API da Santander Dev Week 2023) para obetr os dados de cada cliente
3. Depois de obter os dados dos clientes, você vai usar o API do ChatGPT (OpenAI) para gerar uma mensagem de marketing personalizada para cada cliente. Essa mensagem deve enfatizar a importância dos investimentos.
4. Uma vez que  a mensagem para cada cliente esteja pronta, você vai enviar essas informações de volta para a API, atualizando a lista de 'news' de cada usuário usando o endpoint PUT https://sdw-2023-prd.up.raiway.app/users/{id}
'''

# Utilize sua própria URL se quiser
# Repositório da API: https://github.com/digitalinnovationone/santander-dev-week-2023-api

sdw2023_api_url = 'https://sdw-2023-prd.up.railway.app'

# ETL - Extract
# Extrair a lista de IDs de usuaŕio a partir do arquivo CSV. Para cada ID, faça uma requisição GET para obter os dados do usuário correspondente
# Extract
# https://sdw-2023-prd.up.railway.app/swagger-ui/index.html#/Users%20Controller/create

import pandas as pd
import requests
import json

df = pd.read_csv('/home/matheus/Documentos/MeusProjetos/Data-Science/Projeto_santander/IA_Generativa/SDW2023.csv')

user_IDs = df['UserID'].tolist()
print(user_IDs)


def get_user(id):
  response = requests.get(f'{sdw2023_api_url}/users/{id}')
  return response.json() if response.status_code == 200 else None


users = [user for id in user_IDs if (user := get_user(id)) is not None]

print(json.dumps(users, indent=2))


# ETL - Transform
#Utilize a API do OpenAI GPT-4 para gerar uma mensagem de marketing personalizada para cada usuário
import openai

openai.api_key = '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'

def generate_ai_news(user):
  completion = openai.ChatCompletion.create(
      model = 'gpt-4',
      messages = [
          {
              'role': 'system',
              'content': 'Você é um especialista em markting bancário.'
          },
          {
              'role': 'user',
              'content': f'Crie uma mensagem para o {user["name"]} sobre a importância dos investimentos (máximo de 100 caracteres)'
          }
      ]
  )
  return completion.choices[0].message.content.strip('\"')

for user in users:
  news = generate_ai_news(user)
  print(news)
  user['news'].append({
  "icon": "https://digitalinnovationone.github.io/santander-dev-week-2023-api/icons/cards.svg",
  "description": news
  })

# ETL - Load
# Atualize a lista de 'news' de cada usuário na API com a nova mensagem gerada
# Load

def update_user(user):
  response = requests.put(f'{sdw2023_api_url}/users/{id}', json=user)
  return True if response.status_code == 200 else False

for user in users:
  success = update_user(user)
  print(f'User {user["name"]} updated? {success}!')

