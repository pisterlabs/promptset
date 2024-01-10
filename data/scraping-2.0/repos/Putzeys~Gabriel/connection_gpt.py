import openai
import os

API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY  # Define a chave da API para a biblioteca openai

class OpenAI_API:
    def __init__(self):
        pass

    def conversation(self, messages, max_tokens=600, temperature=0.3):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages[-5:],  # Mantém as 5 últimas mensagens (contexto)
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
        )
        return response.choices[0].message['content'].strip()