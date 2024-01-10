from openai import OpenAI
from dotenv import load_dotenv
import os

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_ORGANIZATION = "OPENAI_ORGANIZATION"

load_dotenv()

client = OpenAI(
    api_key=os.getenv(OPENAI_API_KEY),
    organization=os.getenv(OPENAI_ORGANIZATION),
)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you today?"}
    ]
)

print(completion.choices[0].message)