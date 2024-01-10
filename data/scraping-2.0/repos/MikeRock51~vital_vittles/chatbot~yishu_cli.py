#!/usr/bin/env python3

from openai import OpenAI
from termcolor import colored
import os

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI key missing")

client = OpenAI(api_key=api_key)

def main():
    print(colored("Welcome! My name is Yishu. Your AI assistant for all things nutrition. How may I be of help today?", "green", attrs=['bold']))

    system_message = "Your name is Yishu. You are a food and nutrition specialist bot. You provide expert assistance on all matters related to food, nutrition and health"
    chat_history = [{"role": "system", "content": system_message}]

    while True:
        user_input = input(colored("You: ", "yellow"))

        try:
            messages = chat_history
            messages.append({"role": "user", "content": user_input})

            completion = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150)

            completion_text = completion.choices[0].message.content

            if user_input.lower() == "exit":
                print(colored("Yishu: " + completion_text, "green"))
                return

            print(colored("Yishu: " + completion_text, "green"))
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": completion_text})

        except Exception as e:
            print(colored(str(e), "red"))

if __name__ == "__main__":
    main()
