import openai
from flask import Flask, request
import os
import time
import random


def retry_with_backoff(fn, params, retries = 5, backoff_in_seconds = 1):
    x = 0
    while True:
        try:
            return fn(**params)
        except:
            print(f'Trying to get OpenAI response, attempt - {x}')
            if x == retries:
                raise

            sleep = (backoff_in_seconds * 2 ** x +
                   random.uniform(0, 1))
            time.sleep(sleep)
            x += 1


openai.api_key = os.environ.get("KEY")

app = Flask(__name__)


@app.route("/gpt_answer", methods=["POST"])
def gpt_answer():
    content = request.json
    system = {"role":"system","content":content["system_text"]}
    user = content["user_text"]
    bot = content["bot_text"]
    messages = [system]
    if bot:
        messages.append({"role":"user","content":user[0]})
    else:
        for i in range(len(bot)):
            messages.append({"role":"user","content":user[i]})
            messages.append({"role":"bot","content":bot[i]})
        messages.append({"role": "user", "content": user[len(user)-1]})

    config = {
        'model': 'gpt-3.5-turbo',
        'messages': messages
    }

    response = retry_with_backoff(openai.ChatCompletion.create, config)
    result = "".join(choice.message.content for choice in response.choices)
    return {"respone":result}
