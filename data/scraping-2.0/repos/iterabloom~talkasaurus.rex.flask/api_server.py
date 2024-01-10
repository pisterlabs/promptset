import logging
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import os
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from queue import Queue
import base64
import atexit
import sqlite3
from collections import Counter
from typing import List, Dict
import random
import time
from textblob import TextBlob
from threading import Thread
from itertools import cycle
import pandas as pd
import vaderSentiment.vaderSentiment as vader
from conversational_ai_engine import DevOpsBot

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

from github import Github
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
OWNER = "iterabloom"
REPO_NAME = "talkasaurus.rex.flask"
github = Github(GITHUB_PERSONAL_ACCESS_TOKEN)
repo = github.get_repo(f"{OWNER}/{REPO_NAME}")

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

# Instantiate chat manager and adaptability module
from conversational_ai_engine import ConversationHandler
chat_handler = ConversationHandler()
from conversational_ai_engine import UserAdaptability
adaptability_module = UserAdaptability()

class BufferStream(Queue):
    def __init__(self, buffer_max_size: int = 5):
        super().__init__(maxsize=buffer_max_size)

    def read(self):
        return self.get()

def transcribe_audio_stream(stream):
    client = speech.SpeechClient()
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_speaker_diarization=True,
        diarization_speaker_count=2
    )
    stream_buffer = BufferStream()

    for chunk in stream:
        stream_buffer.put(chunk)
        if stream_buffer.full():
            audio_content = stream_buffer.get_nowait() 
            request = speech.StreamingRecognizeRequest(audio_content=audio_content)
            responses = client.streaming_recognize(config, [request])
            for response in responses:
                print(response)
                
def convert_text_to_speech(text: str):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US",
                                              ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_data = response.audio_content
    socketio.emit('response', {'audio_data': base64.b64encode(audio_data).decode()})

def storeConversationData(conversations):
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY, user TEXT, bot TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    for conversation in conversations:
        cursor.execute('INSERT INTO conversations (user, bot) VALUES (?, ?)', (conversation['user'], conversation['bot']))
    conn.commit()
    #[print(f"User: {dialogue['User']}\nResponse: {dialogue['Response']}") for dialogue in conversations]

def generate_ai_response(user_message: str) -> str:   
    """ 
    TODO: this function does not handle cases of no response being returned from the GPT4 API. 
          The statement `response = None` overwrites the prior `response` assignment and its value is directly accessed 
          on a later line (`return response['choices'][0]['message']['content'] if response else None`). 
          This will throw a `TypeError: 'NoneType' object is not subscriptable` exception if the `response` is `None`.

    Chat with OpenAI's GPT4-32k. 
    The conversation state is managed internally with the help of message history. 
    Each message object in the messages array has three properties: role, content, and filename.
    Increasing the max tokens increases response length up to a limit.
    """
    # Pass the user message through the chat manager and adaptability module
    processed_message = chat_handler.process(user_message)
    adapted_message = adaptability_module.adapt(processed_message)
    conversation = {
        'messages': [{"role": "user", "content": f"{adapted_message}"}]
    }
    
    attempts = 0

    while attempts < 5:
        try:
            response = openai.ChatCompletion.create(model="gpt-4-32k", messages=conversation['messages'], max_tokens=150) 
            # If the API call was successful then break out of the loop
            break
        except Exception as e:
            print(e)
            if attempts < 4:  # if fewer than 4 attempts have been made, retry
                wait_time = (2 ** attempts) + (random.randint(0, 1000) / 1000)
                print(f"Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                attempts += 1
                continue
            else:  # after 4 attempts, stop trying
                print("Unable to connect to the API after several attempts.")
                response = None
    return response['choices'][0]['message']['content'] if response else None

app = Flask(__name__, static_folder='talkasaurus-react/build')
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@socketio.on('message')
def handle_message(data):
    """
    TODO: this function is quite long, housing distinct functionalities within itself, 
          making it difficult to understand. Refactor it into smaller functions

    The handle_message function discerns whether a developer message is instructing the DevOpsBot 
    to generate new code. Following that, the new method will take the message's text, divide it into action and feature sections, 
    and feed them to the new DevOpsBot to write new code. Proposed alterations to the file are implemented in the code below.

    A reminder that DevOpsBot is in the process of learning and has vulnerabilities, developer instructions must be entered 
    in the following format for the bot to understand what is asked of it:

    For new features: "dev_ops_bot: create new python function which sorts list in reverse"
    
    If the message does contain orders for the bot, the bot segment divides the text and nature of the order 
    and transfers it to the bot to conduct the new functionality. The generated code and its respective review may 
    then be simulated or acted upon as required, possibly being delivered back as a response or employed immediately to the repository.

    Although it now matches the bot's requirements, the code here continues to retain the original handle_message 
    functionality of processing normal messages through the GPT4 model and providing feedback to the user.

    It should be noted that the `notify_developer` method in DevOpsBot class and more of its functionality could
    use more fleshing out. For instance, we'd want the bot to run CI/CD operations, perhaps on a testing or staging branch following 
    any sort of merge into that branch. From there, successful build and all-passing tests on a testing/staging branch might automatically
    trigger a production deployment. The bot could then monitor the prod environment for a period of time, roll back on failure/certain types of exceptions, and so on. 
    """
    logging.info("Received new user message.")

    dialogsCollection = []
    user_message = data['message']

    try:
        logging.info("Starting to process user message.")
        
        result = []

        # Check if user message is an instruction for the devops bot
        if "dev_ops_bot:" in user_message.lower():
            # Initialize the bot
            bot = DevOpsBot()
            # Check if a new feature is instructed to be implemented
            if "create new" in user_message.lower() and "function" in user_message.lower():
                # Split out the action and feature description from the message
                action, feature_desc = user_message.split(":")[1].strip().split("which")
                # Currently hardcoded for Python. This would be dynamically changed based on the repo the bot is applied to.
                new_code, review = bot.new_feature(feature_desc, "python")
                result.extend([new_code, " ", *review])

        else:  # Normal flow
            response = generate_ai_response(user_message)
            logging.info("Starting Text-to-Speech operation.")
            convert_text_to_speech(response)
            socketio.emit('response', {'response': response, 'message': user_message})

            dialogsCollection.append({
                "User": result if result else user_message,
                "Response": response
            })

    except Exception as e:
        logging.error(str(e), exc_info=True)

    logging.info("Finished processing current user message.")
    storeConversationData(dialogsCollection)

# route to retrieve conversation history in CSV
@app.route('/api/conversations/csv', methods=['GET'])
def conversations_csv():
  conn = sqlite3.connect('conversations.db')
  df = pd.read_sql_query('SELECT * FROM conversations', conn)
  return df.to_csv(index=False)

# route to retrieve conversation history in HTML
@app.route('/api/conversations', methods=['GET'])
def conversations_html():
  conn = sqlite3.connect('conversations.db')
  df = pd.read_sql_query('SELECT * FROM conversations', conn)
  return df.to_html(index=False)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0"', port=int(os.getenv('PORT', 5000)))