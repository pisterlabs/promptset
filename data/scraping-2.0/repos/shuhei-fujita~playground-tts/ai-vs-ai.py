from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# .envファイルからOpenAI APIキーを読み込む
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI APIの設定
client = OpenAI(api_key=api_key)

# 初期のテーマとAさんの最初の発言
theme = "結婚記念日"
user_a_message = f"テーマ: {theme}\nAさん: 今日は人々が愛するものを作る素晴らしい日です！"


# 会話を生成する関数
def generate_response(user_input):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant user_a_message. Your answer must be 160 characters or less.",
            },
            {
                "role": "system",
                "content": "AI「Aさん」とAI「Bさん」による会話を生成します。",
            },
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content


try:
    # Aさんの最初の発言を出力
    print(user_a_message)

    while True:
        # Bさんの返答を生成
        user_b_message = generate_response(user_a_message)
        print(user_b_message)

        # Aさんの返答を生成
        user_a_message = generate_response(user_b_message)
        print(user_a_message)

        # 少し待機して会話が見やすくなるようにする
        time.sleep(1)

except KeyboardInterrupt:
    print("会話を終了しました。")
