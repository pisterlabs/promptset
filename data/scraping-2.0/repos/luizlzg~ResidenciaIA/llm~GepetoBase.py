import os
import openai
from openai import OpenAI


class ChatGPT:
    def __init__(self, api_key_path: str):

        with open(api_key_path, 'r') as arquivo:
            # Lê o conteúdo do arquivo
            conteudo = arquivo.read()
        openai.api_key = conteudo

        self.messages = []

    def add_message(self, role: str, prompt: str):
        self.messages.append({'role': role, 'content': prompt})

    def reset_messages(self, maintain_context=True):
        if maintain_context:
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def add_context(self, context: str):
        # inserindo o contexto no começo da lista de mensagens
        context = {'role': 'system', 'content': context}
        self.messages = [context] + self.messages

    def get_completion(self, model="gpt-3.5-turbo-0613", temperature=0):
        response = openai.chat.completions.create(
            model=model,
            messages=self.messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content, response
