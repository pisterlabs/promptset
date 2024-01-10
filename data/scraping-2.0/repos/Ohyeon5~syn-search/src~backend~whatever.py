import os

import openai
from dotenv import load_dotenv

load_dotenv()

input_text = "hello world"
client = openai.AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_BASE_URL"),
)
chat_completion = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[{"role": "user", "content": input_text}],
)

print(chat_completion)
