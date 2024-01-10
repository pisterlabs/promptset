import os
from textwrap import dedent

from dotenv import load_dotenv

load_dotenv()
import openai

openai.api_key = openai_key = os.getenv("OPENAI_KEY")
MODEL = "gpt-3.5-turbo"


def main():
    system_prompt = "You're a scary assistant, you tell scary story to scare people for entertainment. "
    user_prompt = "Please tell me a scary story about a pontianak"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )
    print(response)


if __name__ == '__main__':
    main()
