from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

mes=input("write your message: ")

completion = client.completions.create(
    model="gpt-3.5-turbo",
    prompt=[
        {
            "role": "user",
            "content": mes
        }
    ],
    temperature=1,
    max_tokens=256
)

print(completion.choices[0].text)