import os
import json
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv
from transformers import pipeline
from openai import OpenAI
from utils.strings import *
from utils.utils import *
from utils.RoundRobinApiKeys import RoundRobinAPIKeys

app = Flask(__name__)
load_dotenv()

api_keys = [os.environ.get("OPENAI_API_KEY0"),
            os.environ.get("OPENAI_API_KEY1")]

api_key_manager = RoundRobinAPIKeys(api_keys)

story_gen = pipeline("text-generation", "pranavpsv/gpt2-genre-story-generator")


def generateOptionFromChatGpt(paragraph: str, system_content: str):
    try:
        client = OpenAI(api_key=api_key_manager.get_next_key())
        completion = client.chat.completions.create(
            model=GPT_MODEL, messages=gptCompletionMsg(paragraph, system_content)
        )
        response = completion.choices[0].message.content
        return response
    except Exception as err:
        return {KEY_ERROR: str(err)}


# Story generation route
@app.route(ENDPOINT + ROUTE_GENERATE_STORY, methods=[POST])
def generate():
    try:
        # Extract request details
        request_method = request.method
        content_type = request.headers.get(CONTENT_TYPE)

        # Check if the request is a valid POST request with JSON content
        if request_method == POST and content_type == CONTENT_TYPE_JSON:
            data = request.get_json()
            sentence = data.get(KEY_SENTENCE)
            genre = data.get(KEY_GENRE)

            if sentence is None or genre is None:
                return jsonify({KEY_ERROR: MSG_INVALID_PAYLOAD}), 400

            gpt_2 = story_gen(storyGenArg(genre, sentence), max_new_tokens=30, min_length=20)

            gpt_2[0][KEY_GENERATED_TEXT] = cleanGeneratedMsg(
                gpt_2[0][KEY_GENERATED_TEXT]
            )

            initial_paragraph = gpt_2[0][KEY_GENERATED_TEXT]
            scenarios = json.loads(generateOptionFromChatGpt(initial_paragraph, systemContentScenario()))
            gpt_2.append(scenarios)
            return jsonify(gpt_2)
        else:
            return jsonify({KEY_ERROR: MSG_INVALID_METHOD_CONTENT}), 400
    except json.JSONDecodeError as err:
        return jsonify(gpt_2)
    except Exception as err:
        return jsonify({KEY_ERROR: str(err)}), 500

@app.route('/<path:undefined_route>')
def catch_all(undefined_route):
    abort(404)

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
    