from dotenv import load_dotenv
import os
import openai

load_dotenv()

OPENAI_SECRET_KEY = os.getenv('OPENAI_SECRET_KEY')
openai.api_key = OPENAI_SECRET_KEY


def chat_gpt(input, train_data=""):
    print("Sending...")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": train_data},
            {"role": "user", "content": input}
        ]
    )
    print("Done...")

    output: str = completion.choices[0].message
    print(f"{output=}")
    return output


if __name__ == '__main__':
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(completion.choices[0].message)