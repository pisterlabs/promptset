# flaskapp.py
# Text window for interacting for ChatGPT. 
# Shows current chat history and a way to send a new message.

from openai import OpenAI
from flask import Flask, request, render_template, send_from_directory, abort
import os
import tomgpt.helper 
from werkzeug.utils import safe_join

app = Flask(__name__)
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
THREAD = client.beta.threads.create()

@app.route('/')
def chat():
    messages = client.beta.threads.messages.list(
        thread_id=THREAD.id
    )
    return render_template('chat.html', conversation=messages)


@app.route('/process', methods=['POST'])
def process():
    user_input = request.form['user_input']
    message = client.beta.threads.messages.create(
        thread_id=THREAD.id,
        role="user",
        content=user_input
    )
    global ASSISTANT_ID
    run = client.beta.threads.runs.create(
        thread_id=THREAD.id,
        assistant_id=ASSISTANT_ID,
    )
    helper.process_run(run, client, THREAD)
    messages = client.beta.threads.messages.list(
        order="asc",
        thread_id=THREAD.id
    )
    print(messages)
    return render_template('chat.html', messages=messages)

# this was for testing the url read function 
FILES_DIRECTORY = ''
@app.route('/files/<path:filename>')
def get_file(filename):
    try:
        filename = safe_join(FILES_DIRECTORY, filename)
        return send_from_directory(directory=FILES_DIRECTORY, path=filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    ASSISTANT_ID = input('Assistant ID: ')
    app.run()