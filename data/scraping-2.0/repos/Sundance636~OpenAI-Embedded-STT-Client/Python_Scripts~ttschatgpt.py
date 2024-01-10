# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import asyncio
import websockets
import json
import sounddevice as sd
from scipy.io.wavfile import write
import os




openai.api_key = os.environ['API_KEY']

# print(openai.api_key)

duration =  10 # Seconds
sample_rate = 44100 # Hertz

# Initialize the conversation with a system message and an initial user message
conversation = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "You are a helpful AI assistant"}
]

while True:
    # Record audio and wait for the recording to finish
    recording = sd.rec(int(duration * sample_rate), sample_rate, channels=1)
    sd.wait()


    write('output_folder/output.wav', sample_rate, recording) # Save the recording to a .wav file

    #Send the .wav file to OpenAI and output the transcribed response to the console
    audio_file = open("output_folder/output.wav", "rb")
    transcriptJSON = openai.Audio.translate("whisper-1", audio_file)
    transcript = json.dumps(transcriptJSON.text)
    print("You: " + transcript)

    #Delete the audio file
    os.remove("output_folder/output.wav")

    # Add the transcriped text as user input in a gpt conversation
    conversation.append({"role": "user", "content": transcript})

    # Generate a response
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=conversation
    )

    # Get the assistant's message from the response
    assistant_message = response['choices'][0]['message']['content']

    # Print the assistant's message
    print("Assistant: " + assistant_message)

    # Add the assistant's message to the conversation
    conversation.append({"role": "assistant", "content": assistant_message})

'''
#Relay response to microcontroller via websocket
async def hello():
    uri = "ws://192.168.1.78:80/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send(transcript)
        response = await websocket.recv()
        print(response)

asyncio.get_event_loop().run_until_complete(hello())
'''
