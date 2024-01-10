import requests
import json
from flask import Flask, request
import openai
from flask_cors import CORS

openai.api_key = 'YOUR_GPT4_API_KEY'

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the Flask app


@app.route('/get_response', methods=['POST'])
def get_response():
    conversation_history = request.json['conversation']

    # Join all messages in the conversation history into a single string
    # The role and text are separated by a colon and a space ": "
    prompt = '\n'.join([f"{message['role']}: {message['text']}" for message in conversation_history])

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150  # Adjust according to your needs
    )

    return {'response': response.choices[0].text.strip()}


if __name__ == '__main__':
    app.run(port=5000)