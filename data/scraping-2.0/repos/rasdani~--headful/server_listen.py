import os
import json
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import pyaudio
import wave
import time
from openai import OpenAI
from dotenv import load_dotenv
import shutil

load_dotenv()

app = Flask(__name__)
CORS(app)  # Adjust CORS according to your needs

client = OpenAI()

counter = 0

def copy_current_screenshot():
    global counter
    screenshot_dir = f"browser-recordings/screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)
    current_screenshot_path = "browser-recordings/current_screenshot.png"
    # timestamp = str(time.time())
    # copied_screenshot_path = f"{screenshot_dir}/{timestamp}.png"
    copied_screenshot_path = f"{screenshot_dir}/{counter}.png"
    shutil.copy2(current_screenshot_path, copied_screenshot_path)

def add_to_dataset(transcript):
    global counter
    caption = transcript.text
    # caption = "test caption"
    screenshot_dir = f"browser-recordings/screenshots"
    copied_screenshot_path = f"{screenshot_dir}/{counter}.png"
    with open("coordinates.json", "r") as f:
        coordinates = json.load(f)
    
    with open("hint_string.json", "r") as f:
        hint_string = json.load(f)
    
    data = {
        "coordinates": coordinates,
        "hint_string": hint_string,
        "screenshot_path": copied_screenshot_path,
        "caption": caption
    }
    with open("dataset.jsonl", "a") as f:
        json.dump({counter: data}, f)
        f.write('\n')

    counter += 1


    

def transcribe(file_path):
    audio_file= open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcript   

class RecordingThread(threading.Thread):
    def __init__(self, filename, max_duration):
        super().__init__()
        self.filename = filename
        self.max_duration = max_duration
        self.do_record = True

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=160)
        frames = []
        start_time = time.time()
        
        while self.do_record and (time.time() - start_time) < self.max_duration:
            frames.append(stream.read(160, exception_on_overflow=False))

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))

recording_thread = None

@app.route("/bboxes", methods=["POST"])
def handle_bbox():
    copy_current_screenshot()
    global recording_thread
    data = request.get_json()
    # print(data)
    with open("coordinates.json", "w") as f:
        json.dump(data, f)

    if recording_thread is None or not recording_thread.is_alive():
        recording_thread = RecordingThread('browser_talk.wav', 30)  # 60 seconds max duration
        recording_thread.start()

    return jsonify({"message": "Recording started"}), 200

@app.route("/hintString", methods=["POST"])
def handle_hint_string():
    global recording_thread
    data = request.get_json()
    hint_string = data.get('hintString')
    with open("hint_string.json", "w") as f:
        json.dump(hint_string, f)
    
    if recording_thread and recording_thread.is_alive():
        recording_thread.do_record = False
        recording_thread.join(5)  # 5 seconds timeout
    
    transscript = transcribe("browser_talk.wav")
    add_to_dataset(transscript)
    return jsonify({"message": "Received hint code successfully", "hintCode": hint_string}), 200

def test_recording(app):
    with app.test_client() as client:
        print("Simulating /bboxes POST request to start recording...")
        bbox_data = {"example": "data"}  # Replace with actual data you expect
        client.post("/bboxes", data=json.dumps(bbox_data), content_type='application/json')

        input("Recording... Press Enter to stop...")

        print("Simulating /hintString POST request to stop recording...")
        hint_string_data = {"hintString": "example"}  # Replace with actual data
        client.post("/hintString", data=json.dumps(hint_string_data), content_type='application/json')


if __name__ == "__main__":
    app.run(port=5000, threaded=True)
    # test_recording(app)
    # tr = transcribe("browser_talk.wav")
    # print(tr)
    # breakpoint()
