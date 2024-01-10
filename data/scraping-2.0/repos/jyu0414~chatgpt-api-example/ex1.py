import openai

# API Keyを設定
openai.api_key = ""

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello world"}
        ]
    )

print(chat_completion)