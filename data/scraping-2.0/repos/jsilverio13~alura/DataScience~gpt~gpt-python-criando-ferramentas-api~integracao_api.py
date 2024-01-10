import os
import openai
import dotenv

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.system('clear')

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
          "role": "system",
          "content": "Você é um gerador de produtos fictícios e você deve gerar somente o nome dos produtos e a quantidade o usuário solicitar"
        },
        {
            "role": "user",
            "content": "Gere 5 produtos\n"
        }
    ],
    temperature=1.0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["6.", "6)"]
)

print(response)
