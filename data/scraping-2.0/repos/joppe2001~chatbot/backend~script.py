import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Set your OpenAI API key
load_dotenv()

openai.api_key = os.getenv('OPEN_AI_KEY')

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        input_text = request.json.get('input')
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or another appropriate model
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": input_text}]
        )
        return jsonify({'response': response['choices'][0]['message']['content']})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'response': 'Sorry, I am unable to respond right now.'})

if __name__ == '__main__':
    app.run(debug=True)
