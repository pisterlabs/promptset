import openai

# API Keyを設定
openai.api_key = ""

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "しりとりをしてください"},
        {"role": "user", "content": "りんご"}
        ]
    )

print(chat_completion["choices"][0]["message"]["content"])