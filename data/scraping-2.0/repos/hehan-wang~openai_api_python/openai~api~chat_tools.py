import os
from openai import OpenAI

from openai.types.chat import completion_create_params

os.environ['OPENAI_BASE_URL'] = 'https://key.wenwen-ai.com/v1'
os.environ['OPENAI_API_KEY'] = 'sk-6V2exWFBSa2lmuZ7C0D773D1BaEd4fB7A1B6A0A265D550C6'

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    response_format=completion_create_params.ResponseFormat(type="json_object")
)

print(completion.choices[0].message.model_dump_json())
