
from flask import Flask, render_template, Response, stream_with_context
import cv2
import pyaudio
import wave
import openai
from config import OPENAI_API_KEY
from threading import Thread, Event

# Load OpenAI API key
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Set up audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
transcriptions = {}
transcription_count = -1
waiting = False

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

stop_event = Event()

def transcribe_audio(stop_event):
    while not stop_event.is_set():
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if not stop_event.is_set():
                data = stream.read(CHUNK)
                frames.append(data)

        # Write the audio to file
        wf = wave.open("audio.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        t = Thread(target=callWhisper)
        t.start()

def callWhisper():
    with open("audio.wav", "rb") as temp_audio_file:
        global transcription_count
        transcription_count += 1
        local_count = transcription_count
        response = openai.Audio.transcribe("whisper-1", temp_audio_file)
        transcription = response["text"]
        transcriptions[local_count] = transcription
        print(f"Transcribed audio from segment: ")
        print(transcription)

@app.route('/')
def index():
    return render_template('live_subtitle_generation_index.html', transcription='')

def generate_frames():
    while True:
        # Read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(stream_with_context(generate_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-transcription')
def start_transcription():
    global waiting
    if not waiting:
        waiting = True
        stop_event.clear()
        t = Thread(target=transcribe_audio, args=(stop_event,))
        t.start()
    return "waiting..."

@app.route('/stop-transcription')
def stop_transcription():
    print(stop_event)
    stop_event.set()
    print(stop_event)
    print('stopped event')
    return "Stopped transcription."

@app.route('/get-transcription')
def generate_transcriptions():
    global waiting, transcriptions
    content = ''
    cpy = transcriptions
    #while True:
    if len(cpy) > 0:
        
        for i in range(len(cpy)):
            content += cpy[i] + ' <br>'
        waiting = False
    elif not waiting:
        content = "waiting...<br>"
    return content

if __name__ == "__main__":
    app.run(debug=True)


