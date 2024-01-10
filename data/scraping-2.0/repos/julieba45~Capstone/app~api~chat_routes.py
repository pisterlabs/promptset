from flask import Blueprint, jsonify, session, request
from app.models import User, Order, db
from flask_login import current_user, login_user, logout_user, login_required
from app.config import Config
import openai



chat_routes = Blueprint('chat', __name__)

openai.api_key = Config.OPENAI_API_KEY

@chat_routes.route('/', methods=['POST'])
@login_required
def chat():
    try:
        if 'message' not in request.json:
            return jsonify({"error": "Missing message in request"}), 400
        message = request.json['message']
        history = request.json['history']

        #formatting messages FOR AI

        messages = [{"role": item["role"], "content": item["content"]} for item in history]

        messages.insert(0, {"role": "system", "content": "You are a helpful assistant at your online plant store.."})

        messages.append({"role": "user", "content": message})


        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are a helpful assistant."
        #         },
        #         {
        #             "role": "user",
        #             "content": message
        #         }
        #     ],
        #     max_tokens=150
        # )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )

        return jsonify({"response": response['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify(error=str(e)), 500
