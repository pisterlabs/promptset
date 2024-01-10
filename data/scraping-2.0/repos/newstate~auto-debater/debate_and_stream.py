import openai
import time
import json
import os
import subprocess
import requests
import threading
import queue
import logging
from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("/home/pi/auto-debater/config.ini")

# Initialize the OpenAI client
os.environ['OPENAI_API_KEY'] = config_object["USERINFO"]['OPENAI_API_KEY']
client = openai.OpenAI()

# Initialize the queues
text_queue = queue.Queue(maxsize=1)
audio_queue = queue.Queue(maxsize=1)

# pick the debaters. 
# if we're going to create more assistants for all the politicians we'll be making calls to the API to retrieve a list of them
debaters = ['asst_pISJlFY22I5Ic0bVqZSNrJQk', 'asst_e1m8jjbzvNMEV5A4cQAqndKz']
# and their voices from the Elevenlabs API
voices = ['6WKs2YnqPDF4Pf1pc96u', 'uIP7FUfZrEEVXTEGEixB']
# and probably we want their names as well so we should create some kind of json tree

def get_json(obj):
    json_format = json.loads(obj.model_dump_json())
    return json_format

def is_ffmpeg_running():
    # Checks if an ffmpeg process is currently running
    try:
        output = subprocess.check_output(["pgrep", "-f", "ffmpeg"])
        return True
    except subprocess.CalledProcessError:
        # No ffmpeg process found
        return False

def generate_audio(input_text, message_id, assistant_id):
    CHUNK_SIZE = 1024
    # change from if else logic to a for loop once we have a larger group of politicians
    if assistant_id == debaters[0]:
        assistant = voices[0]
    elif assistant_id == debaters[1]:
        assistant = voices[1]
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
    print("Audio generated.")
    # add the generated audio to the queue
    audio_queue.put(f'output_{message_id}.mp3')
    print("Audio added to queue.")

# Set up logging
logging.basicConfig(filename='stream_audio.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

def stream_audio():
    waiting_tune = "short_wait.mp3"
    while True:
        logging.debug(f"Checking audio queue. Queue Empty: {audio_queue.empty()}")
        if audio_queue.empty():
            logging.debug("Queue is empty, checking waiting tune status.")
            if not is_ffmpeg_running():
                logging.debug("Starting waiting tune loop.")
                process = subprocess.Popen([
                    'ffmpeg',
                    '-stream_loop', '-1',
                    '-i', waiting_tune,
                    '-acodec', 'libmp3lame',
                    '-ar', '44100',
                    '-f', 'flv',
                    'rtmp://192.168.1.112:1937/live'
                ], stderr=subprocess.PIPE, text=True)

                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if "error" in line.lower() or "warning" in line.lower():
                        logging.error(line.strip())
            else:
                logging.debug("Waiting tune already playing.")
        else:
            logging.debug("Found audio in queue, attempting to stream.")
            # Stop any existing ffmpeg process
            subprocess.run(["pkill", "-f", "ffmpeg"])

            audio_file = audio_queue.get()
            if os.path.exists(audio_file):
                logging.debug(f"Streaming audio file: {audio_file}")
                process = subprocess.Popen([
                    'ffmpeg',
                    '-re',
                    '-i', audio_file,
                    '-acodec', 'libmp3lame',
                    '-ar', '44100',
                    '-f', 'flv',
                    'rtmp://192.168.1.112:1937/live'
                ], stderr=subprocess.PIPE, text=True)

                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if "error" in line.lower() or "warning" in line.lower():
                        logging.error(line.strip())

                # os.remove(audio_file) # Uncomment to remove audio files after streaming
                audio_queue.task_done()
            else:
                logging.warning(f"Audio file not found: {audio_file}")
                audio_queue.task_done()

        time.sleep(10)

# Start the audio streaming thread
threading.Thread(target=stream_audio, daemon=True).start()

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
                print(f"Processing assistant: {assistant_id}")
                
                if text_queue.empty():
                    print("Creating run and getting message.")
                    latest_message, quotes, message_id = create_run_and_get_message(thread_id, assistant_id)
                    print(f"Latest message: {latest_message}")
                    text_queue.put(latest_message)

                    # write the textual output to a file
                    file.write(f"Message:\n{latest_message}\n")  
                    for quote in quotes:
                        file.write(f"Quote:\n{quote}\n")  
                    file.write("\n")  
                    print("Text written to file and added to text queue.")
                else:
                    print("Waiting for text queue to be processed before generating new message.")

                if not audio_queue.full() and not text_queue.empty():
                    print("Processing text for audio generation.")
                    text_to_convert = text_queue.get()
                    generate_audio(text_to_convert, message_id, assistant_id)
                    text_queue.task_done()
                    print("Audio generation complete and added to audio queue.")
                else:
                    print("Waiting for audio queue to be empty before generating new audio.")

                time.sleep(1)  # To avoid CPU overload and give time for processing
except KeyboardInterrupt:
    print("Stopping the message collection.")