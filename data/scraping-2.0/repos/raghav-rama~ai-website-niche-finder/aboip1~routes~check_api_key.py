from flask import Blueprint, request, jsonify, make_response, json
import os
import openai
import openai.error
from aboip1.views.logging import getLogger
from aboip1.views.rate_limit_handling import completions_with_backoff

logger = getLogger()
bp = Blueprint("check_api_key", __name__)


@bp.route("/check_api_key", methods=["POST"])
def check_api_key():
    try:
        content_type = request.headers["Content-Type"]
        if content_type == "application/json":
            api_key = request.json["apiKey"]
            logger.debug(f"application/json - api_key: {api_key}")
            check_response(api_key)
            return make_response(jsonify({"message": "API key is valid"}), 200)
        else:
            api_key = json.loads(request.data)
            logger.debug(f"no application/json - api_key: {api_key}")
            check_response(api_key)
            return make_response(jsonify({"message": "API key is valid"}), 200)
    except openai.error.AuthenticationError as e:
        logger.exception(f"Incorrect API Key: ")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        return make_response(jsonify({"error": "Invalid API Key"}), 500)
    except Exception as e:
        logger.exception(f"Some error occured")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        return make_response(jsonify({"error": f"Some error occurred, {e}"}), 500)


def check_response(api_key):
    openai.api_key = api_key
    completion = completions_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant"},
            {
                "role": "user",
                "content": f"This is dummy prompt to check whether the provided API is valid or not",
            },
        ],
        n=1,
        temperature=0,
        max_tokens=256,
    )
    logger.debug(f"Test Response from OpenAI: {completion.choices[0].message.content}")
