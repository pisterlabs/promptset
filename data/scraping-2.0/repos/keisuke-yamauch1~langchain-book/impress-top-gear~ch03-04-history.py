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


history = []

while True:
    user_input = read_user_input()
    if user_input == "":
        break
    history.append({
        "role": "user",
        "content": user_input
    })
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        temperature=1,
        max_tokens=2560,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print("GPT: ", response["choices"][0]["message"]["content"])
    print()
    history.append(response["choices"][0]["message"])

    print("=========")
    for i, element in enumerate(history):
        print(f"{i}: {element['role']}: {element['content']}")
