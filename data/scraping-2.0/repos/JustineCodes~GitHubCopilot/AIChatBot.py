# Create chatbot that will take in user input, ask ChatGPT, and return the response.

import openai
import json
import secrets
import my_secrets

openai.api_key = my_secrets.OPENAI_API_KEY

restricted_words = ['password', 'secret', 'confidential']

def chat_with_gpt(message):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message,
        max_tokens=100,
        temperature=0.6,
        n=1,
        stop=None,
        timeout=15,
        )
    reply = response.choices[0].text.strip()
    return reply

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quite', 'exit']:
        print("ChatGPT: Goodbye!")
        break
    else:
        response = chat_with_gpt(user_input)
        print("ChatGPT: " + response)


