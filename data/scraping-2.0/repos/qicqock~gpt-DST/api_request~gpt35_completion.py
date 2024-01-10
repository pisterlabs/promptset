import openai
from config import CONFIG

"""GPT3.5 completion"""

def gpt35_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt_text,
        max_tokens=200,
        temperature=0,
        stop=['--', '\n', ';', '#'],
    )["choices"][0]["text"]