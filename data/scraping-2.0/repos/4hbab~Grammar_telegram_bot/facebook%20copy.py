import os
import tempfile
import textwrap
from typing import List

import openai
import requests
import psycopg2
import speech_recognition as sr
from flask import Flask, request
from happytransformer import HappyTextToText, TTSettings
from pydub import AudioSegment
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer

VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN')
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')

app = Flask(__name__)

def grammar_correction(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Generate a grammar correction of the following sentence:\n\n{input_text}",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()

def paraphrasing(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Generate a paraphrase of the following sentence:\n\n{input_text}",
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].text.strip()

def summarizing(input_text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Summarize the following paragraph:\n\n{input_text}",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# This function splits a paragraph into multiple chunks and calls the model with each chunk
def split_text_into_chunks(text: str, max_chunk_size: int = 250) -> List[str]:
    return textwrap.wrap(text, max_chunk_size, break_long_words=True)


@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        # Facebook Messenger verification
        if request.args.get('hub.verify_token') == VERIFY_TOKEN:
            return request.args.get('hub.challenge')
        else:
            return 'Invalid verification token'
    elif request.method == 'POST':
        data = request.json
        for entry in data['entry']:
            for messaging_event in entry['messaging']:
                if 'message' in messaging_event:
                    sender_id = messaging_event['sender']['id']
                    message = messaging_event['message']

                    if 'text' in message:
                        text = message['text']
                        echo_all(sender_id, text)
                    elif 'attachments' in message:
                        attachments = message['attachments']
                        handle_voice(
                            sender_id, attachments[0]['payload']['url'])

        return 'OK'


def send_message(sender_id, message):
    payload = {
        'recipient': {'id': sender_id},
        'message': {'text': message}
    }
    response = requests.post(
        f'https://graph.facebook.com/v14.0/me/messages?access_token={ACCESS_TOKEN}',
        json=payload
    )
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")


def echo_all(sender_id, input_text):
    first_word = input_text.split()[0].lower()

    if first_word == "correction":
        corrected_text = grammar_correction(input_text)
        send_message(sender_id, corrected_text)

    elif first_word == "paraphrase":
        paraphrased_text = paraphrasing(input_text)
        send_message(sender_id, paraphrased_text)

    elif first_word == "summary":
        summarized_text = summarizing(input_text)
        send_message(sender_id, summarized_text)
    else:
        send_message(
            sender_id, "Please start your message with either 'correction', 'paraphrase' or 'summary'.")


def handle_voice(sender_id, voice_url):
    # Download the voice message file
    response = requests.get(voice_url)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = temp_file.name

    # Save the downloaded file to the temporary location
    with open(file_path, 'wb') as f:
        f.write(response.content)

    # Convert the audio file to the WAV format using pydub
    audio = AudioSegment.from_file(file_path, format="ogg")
    wav_file_path = file_path + ".wav"
    audio.export(wav_file_path, format="wav")

    # Convert the voice file to text using speech_recognition
    r = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)

    # Pass the text to the echo_all function
    echo_all(sender_id, text)

    # Clean up the temporary files
    os.remove(file_path)
    os.remove(wav_file_path)


if __name__ == '__main__':
    app.run()
