import socketio
import struct
import time
import pvporcupine
import pyaudio
import whisper
import numpy as np
import scipy 
import re
import openai
import time
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def getCommand(text):
     response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo",
       messages=[
         {
           "role": "system",
           "content": "You are a machine that gets command in any language and still provide the right command in the following list \n\nstand, walk, forward, backward, left, right, sit, damping, say, none\n\nFor example\n\nsitdown\n-> sit\n\nஎந்துருச்சு நில்லு\n-> stand\n\nबैठ जाओ\n-> sit\n\nகீழே உட்காரு\n-> sit"
         },
         {
           "role": "user",
           "content": "Its a command for robot so give the intent alone here is the command -> "+ text
         }
       ],
       temperature=1,
       max_tokens=256,
       top_p=1,
       frequency_penalty=0,
       presence_penalty=0
     )
     return response.choices[0].message.content

     
# Initialize Socket.io client
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    print("I'm connected to the server!")

@sio.on('disconnect')
def on_disconnect():
    print("I'm disconnected from the server!")

# Connect to Socket.io server
sio.connect('http://localhost:5005')

# Initialize Porcupine (PicoVoice)
porcupine = pvporcupine.create(access_key='FjGUFMbg8DsUo80PHxlI2nO3yVZmcJwlpr1fBhefvm2BIIpItMVA9A==', keyword_paths=["./Pre-on_en_mac_v2_2_0/Pre-on_en_mac_v2_2_0.ppn"])
pa = pyaudio.PyAudio()

audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length)


def record_audio(seconds, rate, frame_length, audio_stream):
    frames = []
    for _ in range(0, int(rate / frame_length * seconds)):
        pcm = audio_stream.read(frame_length, exception_on_overflow=False)
        frames.append(np.frombuffer(pcm, dtype=np.int16))
    return np.concatenate(frames)

try:
    print('Listening for hotwords ...')

    while True:
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Hotword detected!")
            # sio.emit('send_command', "listening")
            print("Recording audio...")
            recorded_audio = record_audio(3, porcupine.sample_rate, porcupine.frame_length, audio_stream)
            # save_audio as .wav file
            print("Saving audio...")
            # using scipy.io.wavfile.write
            scipy.io.wavfile.write("audio.wav", porcupine.sample_rate, recorded_audio)
            # Convert the recorded audio to text using Whisper
            # result = model.transcribe("audio.wav")
            audio_file = open("audio.wav", "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            print(transcript["text"])
            speechText = transcript["text"]
            # convert everything to lowercase
            speechText = speechText.lower()
            # remove all punctuation using regex
            
            speechText = re.sub(r'[^\w\s]', '', speechText)
            # remove all numbers using regex
            
            def findCommand(speechText):
                if "stand" in speechText:
                    return "standUp"
                elif "walk" in speechText:
                    return "walk"
                elif "forward" in speechText:
                    return "goForward"
                elif "backward" in speechText:
                    return "goBackward"
                elif "left" in speechText:
                    return "turnLeft"
                elif "right" in speechText:
                    return "turnRight"
                elif "sit" in speechText:
                    return "sitDown"
                elif "damping" in speechText:
                    return "damping"
                elif "say" in speechText or "greet" in speechText:
                    return "hi"
                else:
                    return "none"
                
            getData = getCommand(speechText)
            print(f"Data: {getData}")
            command = findCommand(getData)
            print(f"Command: {command}")
            sio.emit('send_command', command)

finally:
    if audio_stream is not None:
        audio_stream.close()
    if pa is not None:
        pa.terminate()
    if porcupine is not None:
        porcupine.delete()

# Keep the client running and connected
sio.wait()
