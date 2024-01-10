# Extração: Extrair as perguntas e respostas de um arquivo CSV.

import pandas as pd

def extract_data(file_path):
    data = pd.read_csv(file_path)
    return data

input_file_path = 'faq_data.csv'
data = extract_data(input_file_path)

# Transformação: Preparar os dados para alimentar o modelo de ChatGPT.
# Nenhuma transformação específica é necessária aqui, já que estamos usando perguntas e respostas diretamente.

# Carga: Criar um modelo de ChatGPT e disponibilizá-lo como um serviço de chatbot.

import openai

# Configure sua chave da API do OpenAI

openai.api_key = "SUA_CHAVE_DE_API_AQUI"

def generate_response(question):
    prompt = f"Pergunta: {question}\nResposta:"
    response = openai.Completion.create(
        engine="text-davinci-003",  # Escolha o mecanismo de acordo com suas necessidades
        prompt=prompt,
        max_tokens=50  # Ajuste este valor para controlar o tamanho da resposta
    )
    return response.choices[0].text.strip()

# Exemplo de interação com o chatbot

while True:
    user_input = input("Usuário: ")
    if user_input.lower() == "sair":
        print("Chatbot encerrado.")
        break
    else:
        response = generate_response(user_input)
        print("Chatbot:", response)

# Certifique-se de ter um arquivo CSV chamado "faq_data.csv" contendo perguntas e respostas. Substitua "SUA_CHAVE_DE_API_AQUI" pela sua chave de API do OpenAI.
