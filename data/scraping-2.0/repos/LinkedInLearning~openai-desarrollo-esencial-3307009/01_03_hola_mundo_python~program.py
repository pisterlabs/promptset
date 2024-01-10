import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        { "role":"user", "content": "¡Hola Mundo!" }
    ]
)

print(completion.choices[0].message.content)