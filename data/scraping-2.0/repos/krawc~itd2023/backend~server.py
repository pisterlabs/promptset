from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
import json

app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins

openai.api_key = "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"
model_engine = "text-davinci-002"

@app.route("/api/chat", methods=["POST"])
def chat():
    # Get the prompt from the request data
    print(request.json)
    prompt = request.json["prompt"]

    # Use the ChatGPT model to generate text
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    message = completion.choices[0].message.content

    response = jsonify({"message": message})

        # Set CORS headers
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")


    # Return the response as JSON
    return response

def extract_mp3_url(response):
    events = response.split("\n\n")  # Split events by double newline ("\n\n")

    for event in events:
        lines = event.strip().split("\n")
        event_type = lines[0].replace("event: ", "")
        data_line = lines[1].replace("data: ", "")
        data = json.loads(data_line)

        print(event_type, data)
        if event_type == "completed":
            url = data.get("url")
            if url:
                return url

    return None
@app.route("/api/voice", methods=["POST"])
def voice():
    # Get the text and voice from the request data
    print(request.json)

    text = request.json["text"]["message"]
    voice = request.json.get("voice", "larry")

    # Call the playHT API to convert text to speech
    playHT_url = "https://play.ht/api/v2/tts"
    playHT_headers = {
        "Content-Type": "application/json",
        "AUTHORIZATION": "Bearer 836b46d009054707aff803c5da3d9203",
        "X-USER-ID": "AiI8TMtOClRIwKkOF7lstErKfGK2",
        "accept": "text/event-stream",
        "content-type": "application/json"
    }
    playHT_data = {
        "text": text,
        "voice": voice,
        "speed": 0.8
    }

    playHT_response = requests.post(playHT_url, json=playHT_data, headers=playHT_headers)
    #playHT_response.raise_for_status()
    mp3_url = None

    # Split the response into events
    events = playHT_response.text.split("event: ")

    # Iterate over events in reverse order to find the "completed" event
    for event in reversed(events):
        lines = event.strip().split("\n")
        event_type = lines[0].replace("event: ", "")
        if len(lines) >= 2:
            data_line = lines[1].replace("data: ", "")
            print(event_type, data_line)
            data = json.loads(data_line)
            if 'url' in data_line:
                mp3_url = data.get("url")
                break
   
    response = jsonify({"url": mp3_url}) if mp3_url else jsonify({"message": "MP3 conversion failed."})
    # Set CORS headers
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")

    # Return the playHT API response as JSON
    return response

if __name__ == "__main__":
    app.run()