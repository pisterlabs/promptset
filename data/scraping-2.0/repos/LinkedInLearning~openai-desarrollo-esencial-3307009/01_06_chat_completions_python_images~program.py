import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=key)

completion = client.chat.completions.create(
    max_tokens=200,
    model="gpt-4-vision-preview",
    messages=[
        { "role":"system", "content": "Eres un asistente muy útil." },
        { "role":"user", "content": 
        [
            {"type":"text", "text":"Describe esta imagen."},
            {"type":"image_url","image_url":"https://www.wisdompetmed.com/images/carousel-exoticanimals.jpg"}
        ] }
    ]
)

print(completion.choices[0].message.content)