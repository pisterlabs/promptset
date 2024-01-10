# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import asyncio
import websockets
import json

import sounddevice as sd
from scipy.io.wavfile import write
import os

openai.api_key = os.environ['API_KEY']


duration =  5 # Seconds
sample_rate = 44100 # Hertz

# Record audio
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)

# Wait for the recording to finish
sd.wait()


# Save the recording to a .wav file
write('output.wav', sample_rate, recording)

#Send the audio recording to OpenAI and output the transcribed response to the console
audio_file = open("output.wav", "rb")
transcriptJSON = openai.audio.translations.create(model="whisper-1",file=audio_file)
transcript = json.dumps(transcriptJSON.text)
print(transcript)

#Delete the audio file
os.remove("output.wav")


#Relay response to microcontroller via websocket
async def hello():
    uri = "ws://10.0.0.235:80/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send(transcript)
        response = await websocket.recv()
        print(response)

asyncio.get_event_loop().run_until_complete(hello())
