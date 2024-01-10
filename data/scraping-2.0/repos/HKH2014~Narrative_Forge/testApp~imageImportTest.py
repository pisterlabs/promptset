""" Flask app to run a server """
from flask import Flask, render_template, request, jsonify
import openai
from openai import OpenAI
client = OpenAI()

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = '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'

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    """generate response from gpt-3.5-turbo"""
    user_input = request.form['user_input']

    # Call the Open AI api to generate images
    picture1Resp = openai.Image.create(
        prompt = "", # [ADD FIRST CHOICE HERE TO GAIN IMAGE]
        n = 1,
        size = "200x200"
    )

    picture1 = picture1Resp["data"]
    picture1 = picture1["url"]

    picture2Resp = openai.Image.create(
        prompt = "", # [ADD SECOND CHOICE HERE TO GAIN IMAGE]
        n = 1,
        size = "200x200"
    )

    picture2 = picture1Resp["data"]
    picture2 = picture1["url"]


    # error check for both pictures possibly being missing    
    if not isinstance(picture1Resp, dict):
        print(f"Unexpected response type: {type(response)}")
        return jsonify({'bot_response': 'An error occurred'})

    if not isinstance(picture2Resp, dict):
        print(f"Unexpected response type: {type(response)}")
        return jsonify({'bot_response': 'An error occurred'})

    # bot_response = response.choices[0].message.content

    # You can store the conversation history in a database or in-memory data structure

    return jsonify({'picture1': picture1, 'picture2': picture2})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
