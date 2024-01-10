import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)

completion = client.chat.completions.create(
    model='gpt-3.5-turbo-1106',
    messages=[
        {'role': 'user', 'content': 'create a simple python snippet'}
    ],
    temperature=0
)

print(completion.choices[0].message.content)

