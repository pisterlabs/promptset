
import os
import openai


from dotenv import load_dotenv as ld
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    ld(dotenv_path)

# openai.api_key = "KEY"
# openai.api_key = input()

openai.api_key = os.getenv("api_key")

models = openai.Model.list()
# print(models)


def handle_input(user_input):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_input}])
    return completion

# print(handle_input("user_input").choices[0].message.content)

while True:
    user_input = input("You: ")
    ai_response = handle_input(user_input).choices[0].message.content
    print(ai_response)
