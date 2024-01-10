from flask import Flask, request, jsonify
import openai
from serpapi import GoogleSearch

# Set up your OpenAI API key
openai.api_key = "########"

app = Flask(__name__)

# Define a function to generate a response to a given question using OpenAI
def generate_response(question):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        temperature=0.5,
        max_tokens=1024,
        n=1,
        stop=None,
    )

    message = response.choices[0].text.strip()
    return message


@app.route('/generate', methods=['POST'])
def generate():
    question = request.json['question']
    response = generate_response(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
