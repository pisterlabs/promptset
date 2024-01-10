import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        max_tokens=1,
        messages=[
             {"role": "user", "content": "What is the opposite of up?  Answer in a single word."}
        ]
        )

print(completion.choices[0].message)