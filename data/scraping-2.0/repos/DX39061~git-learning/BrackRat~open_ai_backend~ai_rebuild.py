# 这是使用 ChatGPT 重构后的代码
import openai
from flask import Flask, request
from flask_cors import CORS
import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

openai.api_key = "" # 你的KEY


def api_call(user_input):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=2048,
    )
    logging.info('prompt: %s, text: %s', user_input, response.choices[0].text)
    return response.choices[0].text


app = Flask(__name__)
CORS(app)

@app.route("/process", methods=["GET"])
def process():
    prompt = request.args.get("prompt")
    text = api_call(prompt)
    return {"status": 200, "result": text}


if __name__ == "__main__":
    app.run()