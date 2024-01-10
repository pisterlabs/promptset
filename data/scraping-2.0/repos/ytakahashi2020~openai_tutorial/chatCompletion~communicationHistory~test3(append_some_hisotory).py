from openai import OpenAI
client = OpenAI()


# 会話の開始
messages = [
    {"role": "user", "content": "こんにちは、私は物理学の学生です。現在大学３年生です。子供の頃から物理が好きでした。父も母も物理が大好きで、子供の頃は、祖父に物理について教えてもらいました。また、私には、２歳年下の妹がおり、とても仲が良いです。本日はよろしくお願いします。"}
]

total_tokens_count = 0

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

total_tokens_count += response.usage.total_tokens

# 応答を表示
print("最初の応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append({"role": "user", "content": "私が何を専攻しているか覚えていますか？"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

total_tokens_count += response.usage.total_tokens

# 応答を表示
print("二番目の応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append({"role": "user", "content": "ありがとう。私は、物理学が大好きです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)

total_tokens_count += response.usage.total_tokens

# 応答を表示
print("三番目の応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append({"role": "user", "content": "物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)


# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)

# 必要に応じて会話を続ける
messages.append(
    {"role": "user", "content": "他の物理学についての面白い話をしてください。少し長目が嬉しいです。"})

# APIリクエストを送信
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
total_tokens_count += response.usage.total_tokens

# 応答を表示
print("応答:", response.choices[0].message.content)
print("回答のトークン数:", response.usage.completion_tokens)
print("質問のトークン数:", response.usage.prompt_tokens)
print("全体のトークン数:", response.usage.total_tokens)
print("トークン数の累計:", total_tokens_count)
