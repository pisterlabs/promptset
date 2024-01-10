# importação de bibliotecas
import pandas as pd  # Biblioteca pandas e a renomeia como 'pd' para facilitar o uso.
import requests  # Usada para fazer solicitações HTTP, como GET ou POST.
import json  # Permite trabalhar com dados no formato JSON.
import openai  # Usada para interagir com o GPT-4 da OpenAI.
import logging  # Usada para registrar informações de log durante a execução do programa.


# função que retorna um usuário
def get_user(id):
    response = requests.get(f"{SDW2023_API_URL}/users/{id}")
    return response.json() if response.status_code == 200 else None


# leitura do arquivo
ARQUIVE = "C:\\Users\\wendh\\projects\\bootcamp_santander_python\\02_trilha-python-dio\\SDW2023.csv"

# configuração da API
SDW2023_API_URL = "https://sdw-2023-prd.up.railway.app"

# leitura do arquivo
df = pd.read_csv(ARQUIVE)

# extração dos usuários
user_ids = df["UserID"].tolist()
print(user_ids)

# transformação
users = [user for id in user_ids if (user := get_user(id)) is not None]
print(json.dumps(users, indent=2))

# Documentação Oficial da API OpenAI: https://platform.openai.com/docs/api-reference/introduction
# Informações sobre o Período Gratuito: https://help.openai.com/en/articles/4936830

# Para gerar uma API Key:
# 1. Crie uma conta no OpenAI
# 2. Acesse a seção "API Keys"
# 3. Clique em "Create API key"
# 4. Link direto: https://platform.openai.com/docs/api-keys

# Substitua o texto TODO por sua API Key da OpenAI, ela será salva como uma variável de ambiente
OPENAI_API_KEY = (
    "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"  # virouviral.ofc@gmail.com
)

openai.api_key = OPENAI_API_KEY


def generate_ai_news(user):
    completion = openai.Completion.create(
        engine="text-davinci-002",  # Engine recomendado para textos curtos
        prompt=f"Você é um especialista em marketing bancário.\nCrie uma mensagem para {user['name']} sobre a importância dos investimentos (máximo de 100 caracteres)",
        max_tokens=50,  # Número máximo de tokens na resposta
    )
    return completion.choices[0].text.strip()


for user in users:
    news = generate_ai_news(user)
    print(news)
    user["news"].append(
        {
            "icon": "https://digitalinnovation.one.github.io/santander-dev-week-2023-api/icons/credit.svg",
            "description": news,
        }
    )


def update_user(user):
    response = requests.put(f"{SDW2023_API_URL}/users/{user['id']}", json=user)
    return True if response.status_code == 200 else False


for user in users:
    success = update_user(user)
    print(f"User {user['name']} updated? {success}!")


# Código usado pelo professor na def generate_ai_news

# def generate_ai_news(user):
#     completion = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Você é um especialista em marketing bancário.",
#             },
#             {
#                 "role": "user",
#                 "content": f"Crie uma mensagem para {user['name']} sobre a importância dos investimentos (máximo de 100 caracteres)",
#             },
#         ],
#     )
#     return completion.choices[0].message.content.strip('"')
