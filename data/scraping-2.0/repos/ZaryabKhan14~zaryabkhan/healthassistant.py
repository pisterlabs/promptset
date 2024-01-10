from flask import Flask, render_template, request, redirect, url_for, session, Blueprint
from datetime import datetime
import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("api_key")

# Initialize OpenAI client
client = openai.Client(api_key=api_key)

# Function to handle the chat with the assistant
def assistant_chatbot(user_query, thread_id=None):
    if thread_id is None:
        thread = client.beta.threads.create()
        thread_id = thread.id
    else:
        thread_id = thread_id

    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_query,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id="asst_6o7w7E8I6m0cVfM3zFzePcb9",
        instructions="Provide information related to health queries. Remember, this is not medical advice. For serious health concerns, consult a healthcare professional.",
    )

    # Wait for the run to complete
    while not run.completed_at:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    last_message = messages.data[0]
    response = last_message.content[0].text.value

    return response, thread_id

healthassitant_app = Blueprint('healthassistant', __name__)

# Initialize 'messages' in the session using before_request
@healthassitant_app.before_request
def before_request():
    if 'messages' not in session:
        session['messages'] = []

@healthassitant_app.route('/healthassistant')
def home():
    return render_template('healthassistant.html')

@healthassitant_app.route('/healthassistant', methods=['POST'])
def chat():
    user_query = request.form['query']
    if user_query.strip():
        response, thread_id = assistant_chatbot(user_query, session.get('thread_id'))
        print(thread_id)

        session['thread_id'] = thread_id
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session['messages'].append((timestamp, "User", user_query))
        session['messages'].append((timestamp, "Bot", response))
    return redirect(url_for('healthassistant.home'))


def delete_thread(thread_id):
    if thread_id:
        client.beta.threads.delete(thread_id=thread_id)
@healthassitant_app.route('/clear', methods=['POST'])
def clear_chat():
    thread_id = session.get('thread_id')
    delete_thread(thread_id)  # Delete the thread
    session['messages'] = []
    session.pop('thread_id', None)  # Remove thread_id from session
    return redirect(url_for('healthassistant.home'))
    print(session)
# @healthassitant_app.route('/export', methods=['POST'])
# def export_chat():
#     session['messages'] = []
#     return redirect(url_for('healthassistant.home'))
