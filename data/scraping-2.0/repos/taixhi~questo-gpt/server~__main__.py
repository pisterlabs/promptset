import json
from flask import Flask, render_template, jsonify, request
import openai
import os
from pathlib import Path
from helpers import *

script_location = Path(__file__).absolute().parent
file_location = script_location / 'creds.json'

app = Flask(__name__, 
            static_url_path='', 
            static_folder='client',
            template_folder='templates')

app.config['EXPLAIN_TEMPLATE_LOADING'] = True

# Initialize the OpenAI API with your key.
with open(file_location, "r") as file:
    credentials = json.load(file)


openai.api_key = credentials["OPENAI_API_KEY"]

@app.route('/generate_question', methods=['POST'])
def generate_question():
    write_log("generate_question called with data" + request.form['prompt'])
    # Get the prompt from the client.
    prompt = request.form['prompt']

    # Check if prompt is provided.
    if not prompt:
        return jsonify({"error": "Please provide a prompt."}), 400

    # Generate a question using OpenAI API.
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a very knowledable teacher, who is exceptional at creating challenging but meaningful questions and answers in JSON form, given the prompt, you create quizzes which tests the content in the prompt."},
                {"role": "assistant", "content": f"Here is 10 educational question about: [START PROMPT]{prompt}[END PROMPT] in JSON form [{{question: , answer: answer}}]. The questions are factual, contextual and specific, with a short answer of at most a few words. For each answer, I will quote the reference from the prompt to show where I got my answer from, like this: answer [quote]."}
            ],
            temperature = 0.5
        )
        result = response.choices[0].message.content.strip()
        questions = json.loads(result)
        write_log(f"questions generated for {prompt}: {questions}")
        res = render_template('questions.html', data=questions)
        write_log(str(result) + "returned:" + str(res))
        return res
    except Exception as e:
        write_log(json.dumps({"error": "Internal error with args %s, %s" % (e.args, type(e))}))
        return jsonify({"error": "Internal error. %s" % e}), 500

if __name__ == '__main__':
    app.run(debug=True)
