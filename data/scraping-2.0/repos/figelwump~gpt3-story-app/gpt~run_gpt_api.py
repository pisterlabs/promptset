from http import HTTPStatus
import json
import os
import openai

from flask import Flask, request, Response
from flask_cors import CORS


def run_web_app():
    app = Flask(__name__)
    CORS(app)  # allow CORS for all routes, all domains (NOTE: not production ready)

    openai.api_key = os.getenv('OPENAI_KEY')

    @app.route("/gpt", methods=["POST"])
    def gpt():
        print("in gpt route: ")
        print(request)
        prompt = request.json["prompt"]
        print(prompt)

        res = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=200,
            temperature=0.9,
            top_p=1,
            n=1,
            frequency_penalty=0.8,
            stop="\n"
        )
        print(res)
        for choice in res['choices']:
            if choice['text'].strip() is not "":
                return {'gptResponse': choice['text']}

        # API didn't return any non-empty completions
        return {'gptResponse': ""}

    app.run(port=3001)


#
# start it
#
run_web_app()
