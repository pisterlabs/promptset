#!/usr/bin/env -S poetry run python

from openai import OpenAI

class consome_openAI():

    def __init__(self, nome:str, perfil:str, resultado_invest):
        self.auth = "secreto"
        self.nome = nome
        self.perfil = perfil
        self.resultado_invest = resultado_invest

    def openAI(self):
        client = OpenAI(api_key=self.auth)
        prompt = f"Persona: Você é um especialista em marketing bancário, mas não insira isso na frase. Agora, crie uma mensagem de duas linhas para o cliente {self.nome} com perfil {self.perfil} recomendando os investimentos dentro do seu perfil dentro da lista: {self.resultado_invest}."

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )

        return completion.choices[0].message

        # stream = client.completions.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt=prompt,
        #     temperature=0.7,  # Você pode ajustar a temperatura conforme necessário
        #     max_tokens=70,  # Ajuste este valor conforme necessário para obter o comprimento desejado da frase
        #     stop=None,  # Pode adicionar paradas personalizadas se necessário
        #     stream=True,
        # )
        output = ""
