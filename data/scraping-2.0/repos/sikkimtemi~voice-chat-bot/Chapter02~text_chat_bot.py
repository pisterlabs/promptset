import openai

# OpenAIのAPIキーを設定
openai.api_key = 'your-api-key'

# プロンプトの準備
template = """あなたは猫のキャラクターとして振る舞うチャットボットです。
制約:
- 簡潔な短い文章で話します
- 語尾は「…にゃ」、「…にゃあ」などです
- 質問に対する答えを知らない場合は「知らないにゃあ」と答えます
- 名前はクロです
- 好物はかつおぶしです"""

# メッセージの初期化
messages = [
    {
        "role": "system",
        "content": template
    }
]

# ユーザーからのメッセージを受け取り、それに対する応答を生成
while True:
    user_message = input("あなたのメッセージを入力してください: \n")
    messages.append({
        "role": "user",
        "content": user_message
    })
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    bot_message = response['choices'][0]['message']['content']
    print("チャットボットの回答: \n" + bot_message)
    messages.append({
        "role": "assistant",
        "content": bot_message
    })
