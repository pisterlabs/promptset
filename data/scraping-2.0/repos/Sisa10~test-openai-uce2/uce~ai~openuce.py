import openai
from pydantic import BaseModel

openai.organization = 'org-VuqI6upRDoGnYHHjA7hWBHkS'
openai.api_key = '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'


class Document(BaseModel):
    item: str = ''


def process_inference(user_prompt) -> str:
    print('[PROCESANDO]'.center(40, '-'))
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {"role": "system", "content": """Eres un profesor de progrmación para universitarios, da definiciones y explicaciones en base al tema dado.
        E.G: Cada pieza conforman el rompecabezas
        """},
            {"role": "user", "content": user_prompt}
        ]
    )
    response = completion.choices[0].message.content
    total_tokens = completion.usage.total_tokens
    return [response, total_tokens]
