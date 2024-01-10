import openai
import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    # api_key defaults to os.environ.get("OPENAI_API_KEY")
    api_key= openai.api_key ,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Gere nomes de produtos fictícios sem descrição de acordo com a requisição do usuário"
        },
        {
            "role": "user",
            "content": "Gere 5 produtos"
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion)
