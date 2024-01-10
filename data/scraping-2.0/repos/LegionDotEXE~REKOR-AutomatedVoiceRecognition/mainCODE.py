import base64
import threading
import queue
from flask import Flask, render_template, jsonify, Response
import speech_recognition as sr
import openai

# Set up your OpenAI API key
openai.api_key = 'HERE' #Please set the key to your own OPENAI API key

app = Flask(__name__)

# Initialize the recognizer and microphone
r = sr.Recognizer()
mic = sr.Microphone()

# Create a Queue
transcription_queue = queue.Queue()

# Flag to control audio listening
listening = False

# Function to listen for audio input and transcribe
def listen_and_transcribe():
    global listening

    with mic as source:
        print("Listening...")
        while listening:
            audio = r.listen(source, phrase_time_limit=1)  # Capture audio from the microphone for 1 second

            if not listening:
                break

            try:
                # Convert audio data to Base64
                audio_base64 = base64.b64encode(audio.get_raw_data()).decode('utf-8')

                # Truncate the audio data if needed
                max_audio_length = 8000  # Adjust this value based on the model's context length limit
                truncated_audio_base64 = audio_base64[:max_audio_length]

                # Transcribe the captured audio using the OpenAI API
                print("Sending audio to OpenAI API...")
                response = openai.Completion.create(
                    engine='davinci',
                    prompt="Transcribe the following audio: " + truncated_audio_base64,
                    max_tokens=50,  # Reduced completion length
                    temperature=0.6
                )

                # Check the API response
                print("API Response:", response)

                if response.choices:
                    text = response.choices[0].text.strip()
                    transcription_queue.put(text)  # Put the transcribed text into the queue
                    print("Transcription:", text)  # Print the transcribed text
                else:
                    print("API did not return any transcription.")
            except Exception as e:
                print("Sorry, there was an error:", str(e))


# Function to start listening
@app.route("/start_asr", methods=["GET"])
def start_asr():
    global listening

    if not listening:
        listening = True
        threading.Thread(target=listen_and_transcribe).start()

    return jsonify({"status": "success"})


# Function to get audio transcriptions
@app.route("/get_audio", methods=["GET"])
def get_audio():
    transcription = transcription_queue.get()
    return jsonify(transcription)


# Function to stream transcriptions
@app.route("/stream", methods=["GET"])
def stream():
    def generate():
        while True:
            transcription = transcription_queue.get()
            yield f"data: {transcription}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# Main route
@app.route("/", methods=["GET"])
def index():
    return render_template("Visual.html")


if __name__ == "__main__":
    app.run(debug=True)
