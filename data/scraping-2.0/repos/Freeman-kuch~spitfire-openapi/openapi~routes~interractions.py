"""Openai interactions module"""
from flask import Blueprint, request, session, jsonify
import openai
from openai.error import RateLimitError
from collections import defaultdict
from openapi.utils import get_current_analytics, handle_check_credits
import os

conversation = Blueprint("interraction", __name__, url_prefix="/api/chat")


def generate_chat_completion(message, chat_log) -> str:
    """
    Generates a chat completion using the GPT-3.5-turbo model from OpenAI.

    Args:
        message (str): The user input message for the chat completion.
        chat_logs (List[str]): A list of chat logs containing previous messages.

    Returns:
        str: The content of the generated response as a string.
    """
    messages = [
        {"role": "system", "content": f"{chat_log}"},
        {"role": "user", "content": message},
    ]

    current_analytics = get_current_analytics()
    if current_analytics.openai_requests < int(os.getenv("DAILY_LIMIT")):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.5, max_tokens=200
        )
        current_analytics.openai_requests += 1
        current_analytics.update()
        return response["choices"][0]["message"]["content"].strip("\n").strip()
    return "Daily limit reached, please try again tomorrow"


#  completion route that handles user inputs and GPT-4 API interactions.
@conversation.route("/completions", methods=["POST"])
@handle_check_credits(session)
def interractions(user):
    """
    Process user input using the GPT-3.5-turbo API and return the response as a JSON object.

    :param user: The user object containing information about the current user.
    :return: JSON object with the response from the GPT-3.5 model API
    """
    chat_logs = defaultdict(list)
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        req = request.get_json()
        if "user_input" and "history" not in req:
            return (
                jsonify(
                    {
                        "message": "Invalid request! Missing 'user_input' or 'history' key."
                    }
                ),
                400,
            )
        history = req.get("history")
        user_input = req.get("user_input")
    else:
        return jsonify({"message": "Content-Type not supported!"}), 406

    if not isinstance(history, list) and not isinstance(user_input, str):
        return (
            jsonify(
                {
                    "message": "Invalid data type for 'history' or 'user_input' field. Must be a valid array or string."
                }
            ),
            400,
        )
    converse = chat_logs.__getitem__(user.id)
    converse.clear()
    converse.append(history)

    try:
        result = generate_chat_completion(message=user_input, chat_log=history)
        # converse.append(f"AI: {result}")
        user.credits -= 1
        user.update()
        return jsonify({"message": result}), 201
    except RateLimitError:
        return (
            jsonify(
                content="The server is experiencing a high volume of requests. Please try again later."
            ),
            400,
        )
    except Exception as error:
        return (
            jsonify(content="An unexpected error occurred. Please try again later."),
            500,
        )


@conversation.route("/", methods=["POST"])
@handle_check_credits(session)
def string_completion(user):
    """
    Process user input using the GPT-3.5-turbo API and return the response as a JSON object.

    :param user: The user object containing information about the current user.
    :return: JSON object with the response from the GPT-3.5-turbo model  API
    """
    content_type = request.headers.get("Content-Type")
    if content_type == "application/json":
        req = request.get_json()
        if "user_input" not in req:
            return (
                jsonify({"message": "Invalid request! Missing 'user_input' key."}),
                400,
            )
        user_input = req.get("user_input")
    else:
        return jsonify({"message": "Content-Type not supported!"}), 406

    if not isinstance(user_input, str):
        return (
            jsonify(
                {
                    "message": "Invalid data type for  'user_input' field. Must be a valid string."
                }
            ),
            400,
        )
    messages = [
        {
            "role": "system",
            "content": "you are a very helpful and professional assistant",
        },
        {"role": "user", "content": user_input},
    ]

    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.5, max_tokens=200
        )
        response = result["choices"][0]["message"]["content"].strip("\n").strip()
        user.credits -= 1
        user.update()
        return jsonify({"message": response}), 201
    except RateLimitError:
        return (
            jsonify(
                content="The server is experiencing a high volume of requests. Please try again later."
            ),
            400,
        )
    except Exception as error:
        return (
            jsonify(content="An unexpected error occurred. Please try again later."),
            500,
        )


@conversation.route("/cron", methods=["GET"])
def cron():
    """
    Returns a JSON object with the key "hello" and the value "world".

    Example Usage:
    ```python
    GET /cron
    ```
    """
    return {"hello": "world"}
