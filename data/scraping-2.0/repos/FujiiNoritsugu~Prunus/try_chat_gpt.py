import os
import openai

# ファイルを開く
with open(os.getenv("OPENAI_API_KEY"), "r") as file:
    # ファイルからデータを読み込む
    data = file.read()

# 読み込んだデータを変数に設定
openai.api_key = data

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
        },
        {
            "role": "user",
            "content": "Compose a poem that explains the concept of recursion in programming.",
        },
    ],
)

print(completion.choices[0].message)
