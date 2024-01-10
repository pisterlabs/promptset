import openai
import os

from dotenv import load_dotenv as ld
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    ld(dotenv_path)


# path=os.path.dirname(__file__).replace('//', '\')
# dotenv_path=f'{path}\.env'


openai.api_key = os.getenv("api_key")

models = openai.Model.list()
# print(models)


def handle_input(user_input):
    copletion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    return copletion


# print(handle_input(input()).choices[0].message.content)

while True:
    print(handle_input(input("You: ")).choices[0].message.content)


