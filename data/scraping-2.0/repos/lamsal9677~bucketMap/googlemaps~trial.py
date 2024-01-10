# help.py

from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Add this import
import openai

app = Flask(__name__)
CORS(app)  # <-- This will enable CORS for all routes

openai.api_key = ''

OPENAI_MODEL = "text-davinci-003"

@app.route('/ask-openai', methods=['POST'])
def ask_openai():
    data = request.get_json()
    prompt = data['prompt']

    response = openai.Completion.create(
        model=OPENAI_MODEL,
        prompt=prompt,
        max_tokens=150  # Adjust as needed
    )

    return jsonify(text=response.choices[0].text.strip())

if __name__ == '__main__':
    app.run(port=3000)
