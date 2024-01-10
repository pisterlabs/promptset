import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

knowledge_base = ''
with open('knowledge-base.txt', 'r', encoding="utf-8") as file:
    knowledge_base = file.read()

response = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    temperature=0,
    messages=[
        {
            "role": "user",
            # example for ygenius
            "content": "default instructions here"
        },
        {
            "role": "user",
            "content": knowledge_base
        },
        {
            "role": "user",
            # add your question here
            "content": "question here"
        }
    ],
)

bot_response = response.choices[0].message.content

print(bot_response)
