import os
import openai
import dotenv
from openai import OpenAI


dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    # api_key defaults to os.environ.get("OPENAI_API_KEY")
    api_key= openai.api_key ,
)

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "Você é um categorizador de produtos."
        },
        {
            "role": "user",
            "content": "Escova de dente"
        }
    ]
)
print(chat_completion.choices[0].message)
