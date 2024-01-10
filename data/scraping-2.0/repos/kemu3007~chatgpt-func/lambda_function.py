import json
import os

import openai
import requests
from dotenv import load_dotenv
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

load_dotenv()

DISCORD_APP_ID = os.getenv("DISCORD_APP_ID")


def verify_signature(event):
    try:
        signature = event["headers"]["x-signature-ed25519"]
        timestamp = event["headers"]["x-signature-timestamp"]
        body = event["body"]
    except KeyError:
        raise Exception("Invalid Request")
    verify_key = VerifyKey(bytes.fromhex(os.getenv("DISCORD_PUBLIC_KEY")))
    verify_key.verify(f"{timestamp}{body}".encode(), bytes.fromhex(signature))


def generate_response(body: dict):
    openai.api_key = os.getenv("OPENAI_SECRET_KEY")
    question = body["data"]["options"][0]["value"]
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question},
        ],
    )
    message = openai_response["choices"][0]["message"]["content"].encode().decode()
    print(question, message)
    response = f"Q. {question}\nA. {message}"
    return response


def lambda_handler(event, context):
    request_data = json.loads(event["body"])
    try:
        verify_signature(event)
        callback_url = f"https://discord.com/api/v10/interactions/{request_data['id']}/{request_data['token']}/callback"
        print(
            requests.post(
                callback_url,
                json.dumps({"type": 5}),
                headers={"Content-Type": "application/json"},
            ).text
        )
        print(json.dumps(request_data))
        if request_data["type"] == 1:
            return {"statusCode": 200, "body": {"type": 1}}
        followup_url = f"https://discord.com/api/v10/webhooks/{DISCORD_APP_ID}/{request_data['token']}"
        print(
            requests.post(
                followup_url,
                json.dumps({"content": generate_response(request_data)}),
                headers={"Content-Type": "application/json"},
            ).text
        )
        return {
            "statusCode": 200,
        }
    except BadSignatureError:
        pass
    except Exception as e:
        print(e)
    print(json.dumps(event))
    if request_data["type"] == 1:
        return {"statusCode": 405, "body": "Bad Signature"}
    return {
        "statusCode": 200,
        "body": json.dumps({"type": 4, "data": {"content": "ダメみたい..."}}),
    }
