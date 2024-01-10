import os
import json
from google.cloud import vision
from google.oauth2 import service_account
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

openai.api_key = "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"

SERVICE_ACCOUNT_JSON_STR = os.environ.get("SERVICE_ACCOUNT_JSON_STR")
service_account_info = json.loads(SERVICE_ACCOUNT_JSON_STR)
credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)


@app.route("/hello_world", methods=["POST"])
def hello_world(request):
    image_data = request.files["image"].read()

    image = vision.Image(content=image_data)

    response = vision_client.label_detection(image=image)
    labels = response.label_annotations

    detected_text = " ".join([label.description for label in labels])

    question = generate_question(detected_text)

    return jsonify({"result": question})


def generate_question(input_text):
    messages = [
        {
            "role": "system",
            "content": "あなたは総合物品アドバイザーです。入力されたテキストに関連するサービスや同類の新しい商品の購入検討、必要な知識の提案など、多角的な視野から役に立ちそうな情報提供のクエリを教えてくれる人です",
        },  # You are a comprehensive goods advisor. You provide queries that offer valuable information from a multifaceted perspective, such as related services based on the entered text, considerations for purchasing similar new products, and suggestions for necessary knowledge
        {
            "role": "user",
            "content": f"{input_text}に関連するサービスや商品、使用上の注意点や関連アイテムについての消費者に役に立ちそうなサービスのアイデアや知見を回答として生成してください。提供する文章は簡潔かつ明快である必要があります。なお、回答は「回答１，回答２…」のように配列になるように羅列してください。回答は2～4個つくってください。回答は簡潔かつ情報量が多くなりすぎないように役に立つ部分に要点を絞ってください。なお、回答以外の文章は挿入しないでください。回答のみ文章として生成してください。input_textはすでにわかっているので、input_textにわざわざ言及しないでください。端的にサービスや提案だけを文章として生成してください",
        },  # Please advise on related services/products, usage cautions, and related items. Responses must be concise and clear, within 50 characters.
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message["content"]


if __name__ == "__main__":
    app.run(debug=True)


# functions-framework==3.*
# google-cloud-vision
# google-auth
# openai

# 回答1: コンセントの故障や火災を防ぐために、定期的な点検とメンテナンスを提供するサービスを利用しましょう。

# 回答2: コンセントの電気使用量をリアルタイムでモニタリングするデバイスを使えば、節電につながるだけでなく、安全性も向上します。

# 回答3: コンセント用のコードオーガナイザーを使えば、コードを整理し、絡まりやすい問題を解決することができます。

# 回答4: コンセントに差し込まれたままになるプラグを防ぐための保護カバーを取り付けることで、安全性を高めることができます。
