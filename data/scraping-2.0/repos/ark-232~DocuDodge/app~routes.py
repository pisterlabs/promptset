import openai
from flask import render_template, request
from werkzeug.utils import secure_filename
import os
import configparser
from app import app

# Load API key from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = config['openai']['api_key']

def process_code_with_gpt(code):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Using GPT-4 chat model
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant focused on industry standard code commenting. DO NOT STOP GENERATING UNTIL THE ENTIRE FILE IS COMPLETE, NO PLACEHOLDER TEXT."},
                {"role": "user", "content": f"Based on the langauge the code is written in, Comment and document the following code (ONLY PROVIDE THE COMMENTED CODE IN RESPONSE):\n{code}"},
            ]
        )
        # Assuming the last message will be the response with the comments
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "Error processing code: " + str(e)


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("files[]")
    processed_files = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join('path_to_save_files', filename)
        file.save(file_path)

        with open(file_path, 'r') as f:
            code = f.read()

        comments = process_code_with_gpt(code)
        processed_files.append({
            'original_code': code,
            'comments': comments,
            'filename': filename
        })

    return render_template('display_code.html', processed_files=processed_files)

@app.route('/comment', methods=['POST'])
def comment():
    file_content = request.form['file_content']
    comments = process_code_with_gpt(file_content)
    return comments  # This will just return the comments as a response


# Additional routes and functions as needed
