import openai

from api_key import api_key

openai.api_key = api_key

messages = [{
    "role": "system",
    "content": "유저와 대화를 하는 챗봇입니다."
}]

while True:
    content = input("user: ")
    messages.append({
        "role": "user",
        "content": content
    })

    if content == "exit":
        break

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=2000,
        temperature=1,
        n=1,
    )
    chatgpt_response = chat.choices[0].message.content
    print("chatgpt: ", chatgpt_response)
    messages.append({
        "role": "assistant",
        "content": chatgpt_response
    })
