import openai
from dotenv import load_dotenv

load_dotenv()


def read_user_input():
    user_input = input("USER: ")
    if user_input == "":
        return ""

    while True:
        new_input = input("")
        if new_input == "":
            return user_input
        else:
            user_input += "\n" + new_input


while True:
    user_input = read_user_input()
    if user_input == "":
        break

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=1,
        max_tokens=2560,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print("GPT:", response["choices"][0]["message"]["content"])
    print()
