from openai import OpenAI
client = OpenAI()


# 会話の開始
messages = [
    {"role": "user", "content": "こんにちは。私は物理学の学生です。"},
]

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

# 応答を表示
print("最初の応答:", response.choices[0].message.content)
print("質問のトークン数:", response.usage.prompt_tokens)  # input
print("回答のトークン数:", response.usage.completion_tokens)  # output
print("全体のトークン数:", response.usage.total_tokens)  # total
