import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("MY_KEY"),
)

def ask(query):
    completions = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="gpt-3.5-turbo",
    )
    response = completions['choices'][0]['message']['content']
    
    return response
