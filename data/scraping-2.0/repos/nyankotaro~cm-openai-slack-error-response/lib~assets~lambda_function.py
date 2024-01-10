import json
import os

import boto3
from urllib3 import PoolManager, exceptions

import openai

# パラメータストアから暗号化されたAPI Keyを取得する
ssm_client = boto3.client("ssm")
ssm_response = ssm_client.get_parameter(Name="/openai/api-key", WithDecryption=True)
openai.api_key = ssm_response["Parameter"]["Value"]

# Webhook URLを設定
slack_webhook_url = os.environ["WEBHOOK_URL"]


def lambda_handler(event, context):
    print(event)
    # Slack Event Subscriptionsのチャレンジレスポンス用
    if "challenge" in event["body"]:
        body = json.loads(event["body"])
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"challenge": body["challenge"]}),
        }

    # body内の投稿内容を取得する(プレーンメッセージ)
    json_body_object = json.loads(event["body"])
    if "event" in json_body_object and "text" in json_body_object["event"] and json_body_object["event"]["text"]:
        input_text = json_body_object["event"]["text"]
    else:
        # attachments内の投稿内容を取得する(AWS Chatbotのような加工されたメッセージ)
        input_text = []
        attachment = json_body_object["event"]["attachments"]
        for block in attachment[0]["blocks"]:
            if "text" in block:
                input_text.append(block["text"]["text"])
        input_text = "\n".join(input_text)

    # OpenAIの回答時、またはSlackの再送処理によるイベントの場合処理を行わない
    if "### OpenAIの回答 ###" in input_text or "X-Slack-Retry-Num" in event["headers"]:
        print("do nothing.")
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": "OK"}
    else:
        return_message = generate_openai_response(input_text)
        post_to_slack_webhook(slack_webhook_url, "### OpenAIの回答 ###\n" + return_message)
        print("Sent successfully.")

    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": "OK"}


def post_to_slack_webhook(webhook_url, message):
    http = PoolManager()
    payload = {"text": message}
    headers = {"Content-type": "application/json"}
    try:
        response = http.request("POST", webhook_url, body=json.dumps(payload).encode("utf-8"), headers=headers)
        if response.status != 200:
            raise ValueError(
                f"Request to slack returned an error {response.status}, the response is:\n{response.data.decode('utf-8')}"
            )
    except exceptions.HTTPError as e:
        raise ValueError(f"Request to slack returned an error: {e}")


def generate_openai_response(content):
    engine = "gpt-3.5-turbo"

    completions = openai.ChatCompletion().create(
        model=engine,
        messages=[
            {
                "role": "system",
                "content": "最初にエラー内容を要約してください。",
            },
            {
                "role": "system",
                "content": "エラー内容が解決しそうであれば、解決策を提示してください。解決策の提示が難しい場合は追加情報を求めない形でネクストアクションを提示してください。",
            },
            {
                "role": "system",
                "content": "やり取りは一回で終わらせるようにしてください。",
            },
            {
                "role": "system",
                "content": "なるべく回答は箇条書きにしてください。",
            },
            {
                "role": "system",
                "content": "回答は日本語で返してください。",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    return completions.choices[0].message.content
