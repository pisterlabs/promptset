# Configurações de variáveis de ambiente
from dotenv import load_dotenv

load_dotenv()
from os import environ

#Lib do chatgpt
import openai

openai.api_key = environ['GPT_TOKEN']

class AI:
    def analise_chat_gpt(self, text):
        try:
            chat_input = [
                {'role': 'user', 'content': 'Quero que você comente de maneira cômica o seguinte texto'},
                {'role': 'assistant', 'content': 'Beleza, qual é o texto?'},
                {'role': 'user', 'content': text}
            ]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=chat_input
            )
            assistant_reply = response['choices'][0]['message']['content']
            return assistant_reply
        except:
            print('Erro ao conectar com o ChatGPT.')

