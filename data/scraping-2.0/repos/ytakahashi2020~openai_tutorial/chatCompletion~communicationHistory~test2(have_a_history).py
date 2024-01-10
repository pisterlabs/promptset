from openai import OpenAI
client = OpenAI()

# 単一のリクエストで複数のメッセージを含む
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "こんにちは、私は物理学の学生です。"},
        {"role": "user", "content": "私が何を専攻しているか覚えていますか？"}
    ]
)

print("応答:", response.choices[0].message.content)
