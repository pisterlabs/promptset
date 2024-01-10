import os

import openai
from dotenv import load_dotenv

# key in env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"

chat_completion = openai.ChatCompletion.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": "Write a 2 paragraphs about gran turismo the video game",
        }
    ],
)

print(chat_completion["choices"][0]["message"]["content"])
