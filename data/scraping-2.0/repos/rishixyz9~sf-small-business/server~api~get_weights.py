import os
import openai
import json

openai.organization = "org-bEmILzRGD3s9pC1wat5n6LI0"
openai.api_key = "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"


def get_weights(message) -> json:

    with open("./api/request_template.txt", "r") as f:
        lines = f.read() + f"\n{message}" + "json object:"
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "user", "content": lines}])
        return json.loads(chat_completion["choices"][0]["message"]["content"])