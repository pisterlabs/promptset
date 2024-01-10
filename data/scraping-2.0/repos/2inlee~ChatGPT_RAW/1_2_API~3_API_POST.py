import openai
from flask import Flask, jsonify, request
import os

app = Flask(__name__)


@app.route('/prompt', methods=['POST'])
def generate_answer():
    if request.method == 'POST':
      prompt = request.json['prompt']
      openai.api_key = os.environ.get('API_KEY')
      pre_prompt = "한국어로 친절하게 대답해줘 :)\n\n"
      response = openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": pre_prompt + prompt}
      ],
      max_tokens=3000,
      stop=None,
      temperature=0.5
      )
      answer = response.choices[0].message.content.strip()

      return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001)

