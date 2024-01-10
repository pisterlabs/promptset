# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

from typing import Optional
import time
import chatbots
from chatbots.langchainGPTChatbot import LangChainGPTBot


application = Flask(__name__)
CORS(application)


def get_model(model_name):
        return LangChainGPTBot

chatbots = {}

@application.route("/api/healthy", methods=["GET"])
def healthy():
    return "healthy"

@application.route("/api/create_chat", methods=["POST"])
def create_chat():
    dict_return = {}
    id = request.json.get("id")
    model_name = request.json.get("model_name")
    print(model_name)
    print("Creating chat with id: {}".format(id))
    chatbot = get_model(model_name)(
        prompt_dir="chatbots/prompt_templates",
        original_id=id,
        name="",
    )
    chatbots[id] = chatbot
    # if not original_id or chatbot.history[-1]["sender"] != "AI":
    first_message = (
        chatbot.get_answer()
    )  # get the first answer, as we expect the chatbot to make the first move
    print("created first AI message")
    return first_message
    

@application.route("/api/request_answer", methods=["POST"])
def request_answer():
    current_id = request.json["id"]
    message = request.json["message"]
    chatbot = chatbots[current_id]
    answer = chatbot.get_answer(message)
    return answer


@application.route("/api/set_model", methods=["POST"])
def set_models():
    current_id = request.json["current_id"]
    model = request.json["model"]
    chatbot = chatbots[current_id]
    chatbots[current_id] = get_model(model)(
        history=chatbot.history, original_id=chatbot.original_id, name=chatbot.name
    )
    new_chatbot = chatbots[current_id]
    if len(new_chatbot.history) == 0 or chatbot.history[-1]["sender"] != "AI":
        first_messages = (
            new_chatbot.get_answer()
        )  # get the first answer, as we expect the chatbot to make the first move
        print("created AI message")
        print(first_messages)
    return jsonify({"messages": first_messages})
    # return jsonify({"status": "success"})


if __name__ == "__main__":

    # check if there is a production environment variable
    PRODUCTION = os.environ.get("PRODUCTION", False)
    application.run(debug=(not PRODUCTION))
