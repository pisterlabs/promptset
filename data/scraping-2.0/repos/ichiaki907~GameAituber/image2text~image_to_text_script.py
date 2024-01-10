import base64
import time
import cv2
import openai
import os
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

def encode_image_to_base64(image_path):
    frame = cv2.imread(image_path)  # 画像を読み込む
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode('utf-8')

# .envファイルの内容を読み込む
load_dotenv()

# 画像フォルダの指定
image_folder = "../VideoCapture/frame"

# GPT-4ビジョンモデルへのプロンプト
PROMPT_CONTENT = """
これはVAMPIRESURVIVORのゲーム実況です。
ゲーム実況をする特にキャラ付けのないOLのようなVTuberが読み上げるためのナレーションスクリプトのみを作成してください。
テンションは低めで4行程度でお願いします。
"""

# APIパラメータの設定
api_key = os.environ['API_KEY']

while True:
    # 画像を読み込んでbase64にエンコード
    image_path = os.path.join(image_folder, f"saved_frame_0.jpg")
    base64_image = encode_image_to_base64(image_path)

    print(image_path)

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_CONTENT},  # ここに質問を書く
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},  # 画像の指定の仕方がちょい複雑
            ],
        },
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": api_key,
        "max_tokens": 300,
    }

    # OpenAI APIにリクエストを送信
    start_time = time.time()
    result = openai.ChatCompletion.create(**params)
    end_time = time.time()
    print(result.choices[0].message.content)
    print(f"実行時間（s）: {end_time - start_time} sec")

    # 15秒待機
    time.sleep(2)
