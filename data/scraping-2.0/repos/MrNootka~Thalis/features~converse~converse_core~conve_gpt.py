import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_gpt(model_name, messages):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=messages
    )

    return completion.choices[0].message['content']

def engage_gpt():
    model_choice = input("\nEnter the model: gpt-3.5-turbo-16k or gpt-4: ")
    while model_choice not in ["gpt-3.5-turbo-16k", "gpt-4"]:
        print("\nInvalid model choice. Try again.")
        model_choice = input("\nEnter the model: gpt-3.5-turbo-16k or gpt-4: ")

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        message = input("\nUser: ")
        if message.lower() in ['quit', 'exit', 'q', 'e']:
            break

        messages.append({"role": "user", "content": message})
        response = chat_gpt(model_choice, messages)
        print(f"\n{model_choice}: {response}")
        messages.append({"role": "assistant", "content": response})