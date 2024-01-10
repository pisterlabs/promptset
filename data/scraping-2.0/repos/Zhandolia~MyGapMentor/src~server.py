import os

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

openai.api_key = os.getenv('REACT_APP_OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

@app.route('/api/generate-activities', methods=['POST'])
def generate_activities():
    user_input = request.json['prompt']

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a college application mentor."},
            {"role": "user", "content": user_input}
        ]
    )

    activities = response['choices'][0]['message']['content']
    return jsonify({'activities': activities})

if __name__ == "__main__":
    app.run(debug=True, port=3000)
