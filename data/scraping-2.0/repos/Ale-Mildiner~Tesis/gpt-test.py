#%%
import openai
import os
import requests
import numpy as np
import pandas as pd

key = ""
openai.api_key = os.getenv(key)
#response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)

url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer '
}
data = {
    'model': 'gpt-3.5-turbo',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'I want the answer be a list of python. Sperate in charaters the flowing sentence "Hi how are you?"'}
    ]
}

# Realizar la solicitud a la API
response = requests.post(url, headers=headers, json=data)
result = response.json()

# Obtener la respuesta del modelo
response_text = result['choices'][0]['message']['content']
print(response_text)
#%%
import numpy as np
import pandas as pd

path = "d:/Facultad/Tesis/"
base = pd.read_csv(path+'Corpus_medios_nac.csv', nrows = 10)
notas = base['nota']
notas = list(notas)

def generate_chat_response(prompt, messages):
    openai.api_key = ''  # Replace with your actual API key

    # Format the messages as per OpenAI's chat format
    formatted_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    for idx, msg in enumerate(messages):
        formatted_messages.append({'role': 'user', 'content': msg})
        if idx < len(messages) - 1:
            formatted_messages.append({'role': 'assistant', 'content': ''})

    # Generate a response from the ChatGPT model
    response = openai.Completion.create(
        engine='text-davinci-003',  # Specify the ChatGPT engine
        prompt=formatted_messages,
        temperature=0.7,
        max_tokens=50,
        n=1,
        stop=None,
    )

    # Extract the reply from the response and return it
    reply = response.choices[0].text.strip()
    return reply


def generate_2(message):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': messege}
        ]
    }

    # Realizar la solicitud a la API
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    # Obtener la respuesta del modelo
    response_text = result['choices'][0]['message']['content']
    return response_text

prompt = "Chat with the assistant:"
text = notas[0]
messege = f"Extract the qouted phreasis in this text and give me the answer in a list of python with the qouted phrases: {text}"
response = generate_2(messege)
print(response)

#%%
curl https://api.openai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer " \
  -d '{
    "input": "Your text string goes here",
    "model": "text-embedding-ada-002"
  }'