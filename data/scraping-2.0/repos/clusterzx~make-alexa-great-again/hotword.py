import vosk
import pyaudio
import wave
import time
import threading
from openai import OpenAI
from pathlib import Path
import socketio

sio = socketio.Client()
sio.connect('http://localhost:3010')
model_path = "vosk-model-small-de-0.15"
hotword = "alexa"
model = vosk.Model(model_path)
rec = vosk.KaldiRecognizer(model, 16000)

OPENAI_API_KEY = 'YOUR-API-KEY'

def send_message_to_node_server(action, message):
    if sio.sid:
        sio.emit(action, {'message': message})

class HotwordDetector:
    def __init__(self, callback):
        self.callback = callback
        self.stop_event = threading.Event()

    def detect_hotword(self, stream):
        while not self.stop_event.is_set():
            data = stream.read(1024)
            if rec.AcceptWaveform(data):
                result = rec.Result()
                if hotword in result:
                    self.callback()
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()


def record_phrase(filename, duration=10):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    start_time = time.time()

    while time.time() - start_time < duration:
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def openAI():
    start_time = time.time()
    client = OpenAI(
    api_key=OPENAI_API_KEY,
    )
    audio_file= open("temp_phrase.wav", "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    print("TEXT OUTPUT: " + transcript.text)

    if "male mir ein bild" in transcript.text.lower():
        try:       
            time_difference = time.time() - start_time
            print("DALL-E")
            response = client.images.generate(
            model="dall-e-3",
            prompt=transcript.text,
            size="1024x1024",
            quality="standard",
            n=1,
            )
            image_url = response.data[0].url
            send_message_to_node_server('dall_e', image_url)
            print(image_url)
            print("finished")
            send_message_to_node_server('finished', 'finished')
        except:
            print("DALL-E ERROR")
            send_message_to_node_server('error_output', 'Leider konnte ich kein Bild erstellen. Bitte versuche es erneut.')
            print("finished")
            send_message_to_node_server('finished', 'finished')
    else:
        time_difference = time.time() - start_time
        print("TIME DIFFERENCE OpenAI: " + str(time_difference))
        send_message_to_node_server('text_output', transcript.text)

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer short as possible."},
            {"role": "user", "content": transcript.text}
        ]
        )
        print(completion.choices[0].message.content)
        responseText = completion.choices[0].message.content
        speech_file_path = Path(__file__).parent / "speech.mp3"
        response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=completion.choices[0].message.content
        )

        response.stream_to_file(speech_file_path)
        send_message_to_node_server('response_text', responseText)
        send_message_to_node_server('speech', 'speech.mp3')

        print("finished")
        send_message_to_node_server('finished', 'finished')

def main():
    def hotword_detected_callback():
        print("Hotword detected! Recording phrase...")
        send_message_to_node_server('wake_word_detected', 'Hallo, wie kann ich dir helfen?...')
        record_phrase("temp_phrase.wav")
        print("Speech detected. Recording finished.")
        send_message_to_node_server('rec_stop', 'RECORDING STOPPED')
        openAI()
        send_message_to_node_server('listening', 'Du hast 10 Sekunden Zeit mir deine unbedeutenden Worte mitzuteilen! </br>Ich höre...')

    vosk.SetLogLevel(-1)

    hotword_detector = HotwordDetector(hotword_detected_callback)
    hotword_thread = threading.Thread(target=hotword_detector.detect_hotword, args=(pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024),))
    hotword_thread.start()

    try:
        hotword_thread.join()
    except KeyboardInterrupt:
        hotword_detector.stop()
        hotword_thread.join()

if __name__ == "__main__":
    main()