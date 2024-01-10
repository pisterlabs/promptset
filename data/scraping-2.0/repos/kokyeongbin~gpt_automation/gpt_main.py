import openai

openai.api_key = "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"

messages = []
while True:
    content=input("User: ")
    messages.append({"role":"user", "content":content})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    chat_response = completion.choices[0].messages.content
    print(f'ChatGPT: {chat_response}')
    messages.append({"role":"assistant", "content":chat_response})