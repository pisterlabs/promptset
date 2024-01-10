import openai
from dotenv import load_dotenv
import os

# load .env
load_dotenv()

openai.organization = os.environ.get('OPENAI_ORGANIZATION')
openai.api_key = os.environ.get('OPENAI_API_KEY')

def req_openai(message, temperature=0):
    completion = openai.ChatCompletion.create(
    model=os.environ.get('OPENAI_MODEL'),
    messages=[{"role": "user", "content": message}],
        temperature=temperature,
        max_tokens=int(os.environ.get('OPENAI_MAX_TOKENS')),
    )
    return completion.choices[0]['message']['content'].strip()