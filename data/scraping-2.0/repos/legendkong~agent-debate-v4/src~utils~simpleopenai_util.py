import openai
import os
from src.config.dependencies import *
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_TOKEN')

def simple_output(system_prompt, user_prompt, model='gpt-3.5-turbo', temperature=0):
    response = openai.ChatCompletion.create(
        temperature = temperature,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(user_prompt)}
        ]
    )
    return response['choices'][0]['message']['content']
