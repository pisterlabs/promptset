import openai
import time
import json
import os
import requests
import threading
import queue
from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("/home/pi/auto-debater/config.ini")

# Initialize the OpenAI client
os.environ['OPENAI_API_KEY'] = config_object["USERINFO"]['OPENAI_API_KEY']
client = openai.OpenAI()

# Initialize the queues
text_queue = queue.Queue(maxsize=1)
audio_queue = queue.Queue(maxsize=1)

def get_json(obj):
    json_format = json.loads(obj.model_dump_json())
    return json_format

def generate_audio(input_text, message_id, assistant_id):
    CHUNK_SIZE = 1024
    if assistant_id == 'asst_pISJlFY22I5Ic0bVqZSNrJQk':
        assistant = '6WKs2YnqPDF4Pf1pc96u'
    elif assistant_id == 'asst_e1m8jjbzvNMEV5A4cQAqndKz':
        assistant = 'uIP7FUfZrEEVXTEGEixB'
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{assistant}"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": config_object["USERINFO"]['xi-api-key']
    }

    data = {
    "text": input_text,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    print("Generating audio...")
    response = requests.post(url, json=data, headers=headers)
    print(response.status_code)
    print()
    with open(f'output_{message_id}.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    
    # add the generated audio to the queue
    audio_queue.put(f'output_{message_id}.mp3')

def play_audio():
    while True:
        audio_file = audio_queue.get()  # Wait for an audio file in the queue
        command = f'cvlc --play-and-exit "{audio_file}"'
        os.system(command)        
        audio_queue.task_done()

# Start the audio playing thread
threading.Thread(target=play_audio, daemon=True).start()

# Function to create a run and get the latest message
def create_run_and_get_message(thread_id, assistant_id):
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    print(f"Run created with ID: {run.id}")

    # Wait for the run to complete
    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        time.sleep(5)
        print(f"Checking run status: {run.status}")

    # Retrieve the latest message from the thread
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages_json = get_json(messages)
    latest_message = messages_json['data'][0]['content'][0]['text']['value']
    quotes = [annotation['file_citation']['quote'] for annotation in messages_json['data'][0]['content'][0]['text']['annotations'] if 'file_citation' in annotation]
    print(f"Latest message: {latest_message}")
    print(f"Quotes: {quotes}")

    message_id = messages_json['data'][0]['id']

    return latest_message, quotes, message_id

# IDs of your assistants and the thread
thread = client.beta.threads.create()
get_json(thread)
thread_id = thread.id
print(f"Thread created with ID: {thread_id}")
assistant_ids = ['asst_pISJlFY22I5Ic0bVqZSNrJQk', 'asst_e1m8jjbzvNMEV5A4cQAqndKz']

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Stelt u zichzelf alstublieft voor en noem het onderwerp waarover u wilt beginnen. Houdt het debat alstublieft netjes en respectvol maar wees ook kritisch op elkaar waar nodig.",
)
print(f"Message created with ID: {message.id}")

try:
    with open(f'messages_{thread_id}.txt', 'a') as file:
        while True:
            for assistant_id in assistant_ids:
                if not text_queue.full() and not audio_queue.full():
                    latest_message, quotes, message_id = create_run_and_get_message(thread_id, assistant_id)
                    text_queue.put(latest_message)

                    # write the textual output to a file
                    file.write(f"Message:\n {latest_message}\n")  
                    for quote in quotes:
                        file.write(f"Quote:\n {quote}\n")  
                    file.write("\n")  
                    time.sleep(1)

                if not audio_queue.full() and not text_queue.empty():
                    # Generate audio for the text in the text queue
                    text_to_convert = text_queue.get()
                    generate_audio(text_to_convert, message_id, assistant_id)
                    text_queue.task_done()
                
                time.sleep(1) # Avoid CPU overload

except KeyboardInterrupt:
    print("Stopping the message collection.")
