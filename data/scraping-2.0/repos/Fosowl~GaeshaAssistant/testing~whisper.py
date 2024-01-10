import openai
import os
import signal
import requests
import pyaudio
import wave
from scipy.io.wavfile import write

openai.api_key = os.getenv('OPENAI_KEY') 

# RECORD
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 
CHUNK = 8192
RECORD_SECONDS = 5
p = pyaudio.PyAudio()

killed = False

def get_microphone(wf, stream):
    print("listening...")
    frames = []
    count = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = bytes(0)
        try:
            data = stream.read(CHUNK)
            count += 1
        except:
            pass
        frames.append(data)
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    print("wave writed")
    return count / (RATE / CHUNK)

def handleInterrupt(signum, frame):
    killed = True
    pass

def listen_loop():
    signal.signal(signal.SIGINT, handler=handleInterrupt)
    while not killed:
        wf = wave.open('./record.wav', 'w')
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
        sec = get_microphone(wf, stream)
        wf.close()
        stream.close()
        audio_file = open("./record.wav", "rb")
        if sec > 4:
            print("using whisper...")
            whisper_interpretation = openai.Audio.transcribe("whisper-1", audio_file)
            print(">>> ", whisper_interpretation['text'])
        os.remove('./record.wav')
        sec = 0
    p.terminate()

listen_loop()