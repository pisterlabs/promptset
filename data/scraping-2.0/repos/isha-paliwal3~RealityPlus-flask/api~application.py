import os
from time import sleep
from packaging import version
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import openai
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import json
import base64
import requests
from datetime import datetime

load_dotenv()

# Check OpenAI version is correct
required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ELEVENLABS_KEY = os.getenv('ELEVENLABS_KEY')
voiceID=os.getenv('VOICE_ID')

if current_version < required_version:
  raise ValueError(f"Error: OpenAI version {openai.__version__}"
                   " is less than the required version 1.1.1")
else:
  print("OpenAI version is compatible.")

# Start Flask app
app = Flask(__name__)

origins = [
    "http://localhost:3000",
    "https://reality-plus-web.vercel.app"
]

CORS(app, origins=origins)
client = OpenAI(api_key=OPENAI_API_KEY)

def text_to_speech(textInput, voiceID, elevenLabsApiKey, fileName, stability=None, similarityBoost=None, modelId=None):
    try:
        voiceURL = f'https://api.elevenlabs.io/v1/text-to-speech/{voiceID}'
        stabilityValue = stability if stability else 0
        similarityBoostValue = similarityBoost if similarityBoost else 0

        # Prepare the payload
        payload = {
            "text": textInput,
            "voice_settings": {
                "stability": stabilityValue,
                "similarity_boost": similarityBoostValue
            }
        }

        if modelId:
            payload["model_id"] = modelId

        # Sending the POST request
        response = requests.post(
            voiceURL,
            headers={
                "Accept": "audio/mpeg",
                "xi-api-key": elevenLabsApiKey,
                "Content-Type": "application/json"
            },
            json=payload,
            stream=True  # Important for handling the audio stream
        )

        # Check response status and write to file if successful
        if response.status_code == 200:
            with open(fileName, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            return {"status": "ok", "fileName": fileName}
        else:
            print(f"Error in text-to-speech conversion: {response.status_code}")
            return {"status": "error", "message": f"HTTP Error: {response.status_code}"}

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {"status": "error", "message": str(e)}

def exec_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return str(e)

def lip_sync_message(message):
    time = datetime.now()
    print(f"Starting conversion for message {message}")
    exec_command(f"ffmpeg -y -i audios/message_{message}.mp3 audios/message_{message}.wav")
    print(f"Conversion done in {(datetime.now() - time).total_seconds()} seconds")
    exec_command(f"./bin/rhubarb -f json -o audios/message_{message}.json audios/message_{message}.wav -r phonetic")
    print(f"Lip sync done in {(datetime.now() - time).total_seconds()} seconds")

def read_json_transcript(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def audio_file_to_base64(file):
    with open(file, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')

def createAssistant(client, instructions):
    full_prompt = (instructions + 
                   ''' You will always reply with a JSON array of messages. With a maximum of 3 messages. 
                   Each message has a text, facialExpression, and animation property. 
                   The different facial expressions are: smile, sad, angry, funnyFace, and default. 
                   The different animations are: Talking, Greeting and Idle.''')
    assistant = client.beta.assistants.create(
        instructions=full_prompt,
        model="gpt-4-1106-preview",
    )
  
    assistant_id = assistant.id
  
    return assistant_id

@app.route('/createAssistant', methods=['POST'])
def create_assistant():
    data = request.json
    instructions = data.get('instructions', '')

    if not instructions:
        return jsonify({"error": "Missing instructions"}), 400

    assistant_id = createAssistant(client, instructions)
    return jsonify({"assistant_id": assistant_id})

@app.route('/start', methods=['POST'])
def start_conversation():
  data = request.json
  assistant_id = data.get('assistant_id')

  if not assistant_id:
    return jsonify({"error": "Missing assistant_id"}), 400

  thread = client.beta.threads.create()
  return jsonify({"thread_id": thread.id})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json

    def generate(data):
        thread_id = data.get('thread_id')
        assistant_id = data.get('assistant_id')
        user_input = data.get('message', '')

        if not thread_id:
            yield f"data: Error: Missing thread_id\n\n"
            return

        print(f"Received message: {user_input} for thread ID: {thread_id}")

        client.beta.threads.messages.create(thread_id=thread_id,
                                            role="user",
                                            content=user_input)

        run = client.beta.threads.runs.create(thread_id=thread_id,
                                              assistant_id=assistant_id)

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id,
                                                          run_id=run.id)
            print(f"Run status: {run_status.status}")
            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                response = messages.data[0].content[0].text.value
                yield f"{response}\n\n"
                break
            sleep(1)

    return Response(generate(data), mimetype='text/event-stream')

@app.route('/talk', methods=['POST'])
def talk():
    data = request.json

    def generateTalk(data):
        thread_id = data.get('thread_id')
        assistant_id = data.get('assistant_id')
        user_input = data.get('message', '')

        if not thread_id:
            yield f"data: Error: Missing thread_id\n\n"
            return

        print(f"Received message: {user_input} for thread ID: {thread_id}")

        client.beta.threads.messages.create(thread_id=thread_id,
                                            role="user",
                                            content=user_input)

        run = client.beta.threads.runs.create(thread_id=thread_id,
                                              assistant_id=assistant_id)

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            print(f"Run status: {run_status.status}")
            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                message_content = messages.data[0].content[0].text.value
                message_content = message_content.replace('```json\n', '').replace('\n```', '')

                try:
                    response_data = json.loads(message_content)
                except json.JSONDecodeError:
                    print("Invalid JSON response:", message_content)
                    yield f"data: Error: Invalid response received\n\n"
                    break

                response_messages = []
                for i, message in enumerate(response_data):
                    # Assuming the structure of each message in response_data is similar to /chat
                    text_input = message['text']
                    tts_response = text_to_speech(text_input, voiceID, elevenLabsApiKey, f'audios/message_{i}.mp3')
                    
                    if tts_response['status'] == 'ok':
                        lip_sync_message(i)  # Function to generate lip sync data
                        message['audio'] = audio_file_to_base64(f'audios/message_{i}.mp3')
                        message['lipsync'] = read_json_transcript(f'audios/message_{i}.json')
                        response_messages.append(message)
                    else:
                        print(f"Error in text-to-speech conversion: {tts_response['message']}")

                yield json.dumps(response_messages) + '\n\n'
                break
            sleep(1)
    
    return Response(generateTalk(data), mimetype='text/event-stream')

# Run server
if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True)

