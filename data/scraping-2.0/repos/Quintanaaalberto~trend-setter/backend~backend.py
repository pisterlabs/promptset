from flask import Flask, request, jsonify
import openai
from flask_cors import CORS
import os
import tempfile

openai.api_key = os.getenv('OPENAI-API-KEY')

app = Flask(__name__)
CORS(app)


def api_call(audio_file):
    response = openai.Audio.transcribe(model="whisper-1", file=audio_file)

    # Assuming the response is a dictionary containing a 'text' field
    transcript = response['text']
    return transcript


def call_chatgpt(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    chatgpt_answer = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return chatgpt_answer.choices[0].message["content"]


def summarize(text):
    chatgpt_prompt = f"summarize the following text: {text}"
    return call_chatgpt(chatgpt_prompt)


@app.route('/api/run_script_transcribe', methods=['POST'])
def run_script_transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Create a temporary file to store the uploaded file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp3')
    file.save(temp_path)

    # Call your Python script here using the temporary file
    result = api_call(open(temp_path, "rb"))

    # Close the file descriptor and remove the temporary file
    os.close(temp_fd)
    os.remove(temp_path)

    return jsonify({'transcript': result})


@app.route('/api/run_script_summarize', methods=['POST'])
def run_script_summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Create a temporary file to store the uploaded file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp3')
    file.save(temp_path)

    # Call your Python script here using the temporary file
    result = summarize(api_call(open(temp_path, "rb")))

    # Close the file descriptor and remove the temporary file
    os.close(temp_fd)
    os.remove(temp_path)

    return jsonify({'summary': result})


if __name__ == '__main__':
    app.run(port=5000)
