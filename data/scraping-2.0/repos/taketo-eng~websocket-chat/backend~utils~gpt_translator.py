import openai
from config import settings

openai.api_key = settings.OPENAI_API_KEY

def translate(message, to_lang):
    prompts = [
        {"role": "system", "content": f"You are an assistant that recognises message language and translates to {to_lang}."},
        {"role": "user", "content": f"Please translate to {to_lang} this message and output only plaintext."},
        {"role": "user", "content": f'"{message}"'}
    ]

    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompts,
        n=1
    )

    return completion.choices[0].message.content


