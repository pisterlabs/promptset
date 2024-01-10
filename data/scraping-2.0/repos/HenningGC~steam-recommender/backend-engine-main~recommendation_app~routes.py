
from flask import Blueprint, request, jsonify
#import generate_recommendations
from .recommendation_model import generate_recommendations
import openai
import json
import pandas as pd
import os

main = Blueprint('main', __name__)

@main.route('/recommendations', methods=['POST'])
def get_recommendations():
    data = request.json
    input_data = int(data.get('input', ''))

    try:
        recommendations = generate_recommendations(input_data)
    except Exception as e:
        print(e)
        return jsonify({'error': "User not Found"}), 400

    return jsonify({
        'recommendations': recommendations
    })

@main.route("/chat", methods=["POST"])
def chat():
    openai.api_key = "YOUR_API_KEY_HERE"
    data = request.get_json()
    if not data or "message" not in data or "history" not in data:
        return jsonify({"error": "Message or history not provided"}), 400

    message = data["message"]
    history = data["history"]
    print(history)
    games = pd.read_pickle(r'backend-engine-main\recommendation_app\data\recommendations.txt')
    games_str = ', '.join(games)

    # Initial prompt
    initialPrompt = f"You are an expert in videogames and have vast knowledge about the following games: {games_str}. Your next answer should only say Hello! Would you like to know more about these games?"

    # Parse history string into a list of message objects
    history_lines = history.split('\n')[1:]
    parsed_history = [{"role": "user" if i % 2 == 0 else "system", "content": line[6:]} for i, line in enumerate(history_lines)]

    # Combine the context with the user's message
    messages = [{"role": "user", "content": initialPrompt}]
    messages.extend(parsed_history)
    messages.append({"role": "user", "content": message})

    # Interact with the GPT-3.5 API
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
            messages=messages
        )
    '''
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # this is "ChatGPT" $0.002 per 1k tokens
            messages=messages
        )
        print(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    '''

    chatbot_response = response.choices[0].message.content
    print(chatbot_response)
    return jsonify({"response": chatbot_response})
