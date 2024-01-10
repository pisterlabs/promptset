from  openai import OpenAI
from dotenv import load_dotenv
import os


# API REQUEST ARE LIMITED, DO NOT USE

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def chat_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "you are a code interpreter for my parsed language"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        response = chat_with_gpt(user_input)
        print("GPT-3: ", chat_with_gpt(user_input))