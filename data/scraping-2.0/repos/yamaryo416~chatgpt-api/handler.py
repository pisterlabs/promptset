import json
import os

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def chatgpt(event, context):
    body = json.loads(event["body"])
    if "user" not in body:
        return {
            'statusCode': 400,
            'headers': {
                "Access-Control-Allow-Headers": "Content-Type",
            },
            'body': "Arguments Error: user is required"
        }
    messages = []
    messages.append({"role": "user", "content": body["user"]})

    if "system" in body:
        messages.append({"role": "system", "content": body["system"]})

    if "assistant" in body:
        messages.append({"role": "assistant", "content": body["assistant"]})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
    )

    return {
        'statusCode': 200,
        'headers': {
            "Access-Control-Allow-Headers": "Content-Type",
        },
        'body': json.dumps({"text": response["choices"][0]["message"]["content"]})
    }
