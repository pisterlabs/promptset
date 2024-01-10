import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
print("Welcome to the chat window! Happy chatting! Type 'stop' to end the conversation.\n")
messages = [{"role": "system", "content": "You are a kind helpful assistant"}, ]
while True:
    mes = input("User: ")
    if mes:
        if mes == "stop":
            break
        messages.append({"role": "user", "content": mes},)
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat["choices"][0]["message"]["content"]
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})
