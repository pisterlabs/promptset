import os
import openai
from dotenv import load_dotenv, find_dotenv
from helper import get_completion_from_messages

_ = load_dotenv(find_dotenv())

api_key = os.environ["OPENAI_API_KEY"]
print(f"OpenAI API version {openai.__version__}.lida com sucesso!", )
print(f"Access API Key lida com sucesso!")

mensagens = [{"role": "user", "content": "Quem descobriu o Brasil? Mostre apenas o nome do descobridor."}]
respostas = get_completion_from_messages(api_key, mensagens)
print(respostas)
