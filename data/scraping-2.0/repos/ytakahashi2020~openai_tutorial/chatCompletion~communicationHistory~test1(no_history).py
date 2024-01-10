from openai import OpenAI
client = OpenAI()

# 最初のリクエスト
response1 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "こんにちは、私は物理学の学生です。"}
    ]
)

print("最初の応答:", response1.choices[0].message.content)

# 二番目のリクエスト（前のリクエストの内容を参照せず）
response2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "私が何を専攻しているか覚えていますか？"}
    ]
)

print("二番目の応答:", response2.choices[0].message.content)
