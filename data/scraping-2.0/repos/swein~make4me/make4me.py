#!/usr/bin/env python3

from flask import Flask, render_template, request
import openai

# Set up OpenAI API credentials
openai.api_key = ''

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        requirements = {
            'Goal': request.form['goal'],
            'Input': request.form['input'],
            'Output': request.form['output'],
        }
        code_response = generate_code(requirements)
        return render_template('result.html', code_response=code_response)
    return render_template('index.html')

def generate_code(requirements):
    # Construct the prompt from the requirements
    prompt = construct_prompt(requirements)

    # Get the API response from ChatGPT
    response = get_chat_response(prompt)

    # Extract the code from the API response
    code = extract_code(response)

    return code

def construct_prompt(requirements):
    prompt = f"Goal: {requirements['Goal']}\n\n"
    prompt += f"Input:\n{requirements['Input']}\n\n"
    prompt += f"Output:\n{requirements['Output']}\n\n"

    return prompt

def get_chat_response(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None,
        timeout=30
    )
    return response.choices[0].text.strip()

def extract_code(response):
    return response

if __name__ == '__main__':
    app.run(host='localhost', port=8787)
