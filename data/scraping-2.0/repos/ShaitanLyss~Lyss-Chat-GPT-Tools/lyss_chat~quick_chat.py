from ast import mod
from openai import OpenAI


import sys
from dotenv import load_dotenv
import os
import argparse


def chat_with_gpt(message):
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Make short straight to the point answers."},
            {"role": "user", "content": message},
        ],
        stream=True,
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            sys.stdout.write(content)
            sys.stdout.flush()
    print()


def main():
    parser = argparse.ArgumentParser(description="Chat with GPT")
    parser.add_argument("message", type=str, help="The message to chat with GPT", nargs="+")
    args = parser.parse_args()

    chat_with_gpt(' '.join(args.message))


if __name__ == "__main__":
    main()
