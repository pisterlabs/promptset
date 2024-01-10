from flask import Flask, request, jsonify
import requests

OPENAI_API_KEY = "Your Api key"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/engines/davinci/completions"

app = Flask(__name__)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    if not request.json or 'prompt' not in request.json:
        return jsonify({'error': 'Invalid input'}), 400
    
    prompt = request.json['prompt']

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "max_tokens": 50  # Adjust as needed
    }

    response = requests.post(OPENAI_API_ENDPOINT, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        if 'choices' in response_data:
            answer = response_data['choices'][0]['text'].strip()
            return jsonify({'response': answer}), 200
        else:
            return jsonify({'error': 'Unable to extract valid response from OpenAI'}), 500
    else:
        return jsonify({'error': f'OpenAI API error - {response.content}'}), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
