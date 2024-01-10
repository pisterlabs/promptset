import os
import openai

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_openai_response(input_text: str):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=input_text,
        max_tokens=50,  # Ajuste conforme necessário
        temperature=0.7,  # Ajuste conforme necessário
    )
    return response
