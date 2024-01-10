from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "system", "content": "你是个乐于助人的助手。"},
        {"role": "user", "content": "谁赢得了2020年的世界大赛？"},
        {"role": "assistant", "content": "洛杉矶道奇队在2020年赢得了世界大赛。"},
        {"role": "user", "content": "在哪里比赛的？"},
        {"role": "assistant", "content": "2020年的世界大赛在美国的德州阿灵顿的环球生命体育场举行。"},
        {"role": "user", "content": "这次比赛的关注点在地方？"}
    ]
)

print(response.choices[0].message.content)
