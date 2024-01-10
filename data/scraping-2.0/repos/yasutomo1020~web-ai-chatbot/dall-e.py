# 必要なPythonモジュールをインポート
import os
import openai
from PIL import Image
from io import BytesIO
from base64 import b64decode
import datetime


# OpenAI APIまたは秘密鍵を設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# 生成したい画像を表すプロンプトを入力
prompt_text = input("prompt: ")

# 現在時刻を取得
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# DALL-E APIを呼び出して新しい画像を生成
response = openai.Image.create(
    prompt=prompt_text,  # ここで希望の画像を表すプロンプトを指定
    n=1,
    size="256x256",
    response_format="b64_json"
)

if response == None or 'error' in response:
    raise Exception(response['error'])
else:
    print("success image generate")
# 返ってきたbase64文字列を使って画像を取得
image_b64 = response['data'][0]['b64_json']
image_data = b64decode(image_b64)

# imageフォルダを作成　すでにある場合は作成しない
os.makedirs("image", exist_ok=True)

# PILを使って画像データを読み込み、保存
image = Image.open(BytesIO(image_data))
image.save("image/output_"+str(now)+".jpg")

# 保存した画像を表示
image.show()
