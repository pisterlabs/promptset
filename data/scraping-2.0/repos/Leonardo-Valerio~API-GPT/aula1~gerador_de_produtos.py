import openai
import dotenv
import os
dotenv.load_dotenv()
openai.api_key = os.getenv("API_GPT")
resposta = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {
            "role": "system",
            "content": "Gere nomes de produtos ficticios sem descrição de acordo com a requisição do usuario."
        },
        {
            "role": "user",
            "content": "Gere 5 produtos"
        }
    ]
)
print(resposta)