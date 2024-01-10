import os
from dotenv import load_dotenv
import openai

# environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')

## code starts here

def get_chat_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    message = response['choices'][0]['message']['content']
    return message

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

user_input = input("You: ")
messages.append({"role": "user", "content": user_input})

bot_response = get_chat_response(messages)
print(f"Bot: {bot_response}")
