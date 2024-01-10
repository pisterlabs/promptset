import requests
import os
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import openai
import re
import csv
import json


config_file = 'actions/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)

os.environ['OPENAI_API_KEY'] = config['api_key']
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

empty_thread = openai.beta.threads.create()
thread_id = empty_thread.id
asst_id = config['assistant']


def send_message_to_rasa(text_message):
    url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {"sender": "user", "message": text_message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    return None

def text_to_speech(text):
    # Set your OpenAI API key here
    api_key = os.environ['OPENAI_API_KEY']
    # Set the endpoint URL
    url = "https://api.openai.com/v1/audio/speech"
    # Set the headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Set the data payload
    data = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy"
    }

    # Make the POST request to the OpenAI API
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Play the audio in the notebook
        with open("speech.mp3", "wb") as f:
            f.write(response.content)
        os.system("afplay " + "speech.mp3")
        
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None

#txt_file_path = 'conversation_history.txt'

name = input("Name of participant: ")
date = input("Date of user test: ")

txt_file_path = f'/Users/phildan/Dev/Navel/user_tests/conversation_history_{name}.txt'

# Function to append a single line to the txt file
def append_to_txt(file_path, line):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(line + '\n')

# Create the txt file if it does not exist
if not os.path.isfile(txt_file_path):
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write("Participant: " + name + '\n')
        file.write("Date: " + date + '\n')
else:
    with open(txt_file_path, 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write("Participant: " + name + '\n')
        file.write("Date: " + date + '\n')



while True:

    #record your voice and save it
    print("RECORDING")
    fs = 44100  # Sample rate
    seconds = 6  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("STOP RECORDING")
    write('output.wav', fs, myrecording)
    audio_file = open("output.wav", "rb")
    user_message = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file,
      language='en')
    user_message = user_message.text
    print("You: " + user_message)

    append_to_txt(txt_file_path, "User: " + user_message)
        
    if user_message.lower() == 'stop.' or user_message.lower() == 'stop':
        break
    bot_responses = send_message_to_rasa(user_message)
    if bot_responses:
        for response in bot_responses:
            if 'text' in response:
                print("Bot:", "." )
                pattern = r"【.*$"
                cleaned_response = re.sub(pattern, '', response['text'])
                append_to_txt(txt_file_path, "Bot: " + cleaned_response)
                audio_file = text_to_speech(cleaned_response)


   