from openai import OpenAI
import os


# openai.api_key  = os.getenv('openai.api_key')
client = OpenAI(
    api_key = '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',  # replace with your own API key
)

def TextInput(text):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a language translator, skilled in translate english language to any spoken language"},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content

text = input('Enter your question:')
res = TextInput(text)
print(f'The answer is: {res}')