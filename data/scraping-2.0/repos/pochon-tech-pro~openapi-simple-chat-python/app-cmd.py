import openai
import configparser
import time

config = configparser.ConfigParser()
config.read("config.ini")
openai.api_key = config["openai"]["api_key"]

# エンジンの種類
engine = "gpt-3.5-turbo"

# API待機時間(秒)
wait_sec = 3


# 逐次出力版
def get_response_stream(prompt, engine):
    responses = openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # ストリームアクセス
    )

    # ストリーミング処理
    message = ""
    for response in responses:
        chunk = response["choices"][0]["delta"].get("content")

        if chunk == None:
            pass
        else:
            print(chunk, end="", flush=True)  # 逐次出力
            message += chunk

    return message.strip()


while True:
    # https://qiita.com/yutaka-tanaka/items/d5acbe6beea79b7ffef7
    user_input = input("ChatGPTへの質問入力(Enterで終了):\n")
    if not user_input:
        break

    # 逐次出力
    print("\n↓↓↓ ChatGPTの回答を逐次出力(ctrl+cで強制終了) ↓↓↓\n")

    get_response_stream(user_input, engine)

    print("\n\n↑↑↑ ChatGPTの回答を逐次出力 ↑↑↑\n")

    # API待機時間
    time.sleep(wait_sec)
