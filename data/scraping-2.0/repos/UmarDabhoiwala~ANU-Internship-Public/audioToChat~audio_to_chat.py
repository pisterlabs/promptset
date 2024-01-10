import openai
import pyaudio
import wave
from playsound import playsound
import requests
import json
from halo import Halo
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


OPENAI_API_KEY = config.OPENAI_API_KEY
ELEVENLABS_API_KEY = config.ELEVENLABS_API_KEY

openai.api_key = OPENAI_API_KEY
spinner = Halo(text="Loading", spinner="dots")


headers = {
    "accept": "audio/mpeg",
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json",
}

headers_voices = {
    "accept": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY,
}


response_voices = requests.get(
    "https://api.elevenlabs.io/v1/voices", headers=headers_voices
)
voice_content = (response_voices.content).decode("utf-8")
data_dict = json.loads(voice_content)

name_id_dict = {}
for x in data_dict["voices"]:
    name = x["name"]
    voice_id = x["voice_id"]
    name_id_dict[name] = voice_id


def return_choices():
    return name_id_dict.keys()


def record_audio(filename, duration=5, channels=1, rate=44100, chunk=1024):
    # Initialize PyAudio object
    p = pyaudio.PyAudio()

    # Open a streaming stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("Recording...")

    # Record audio for the specified duration
    frames = []
    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    p.terminate()

    # Save the recorded audio as a .wav file
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()


def transcribe(audio):

    try: 
        audio_file = open(audio, "rb")
        transcript = (openai.Audio.transcribe("whisper-1", audio_file))["text"]
    except:
        transcript = "Hi How are you?"

   

    if len(transcript) < 1:
        transcript = "I can't think of anything to say I'm too shy"

    return transcript


def get_assistant_response(messages, user_query):

    spinner.start("Generating Assistant Response")

    messages.append({"role": "user", "content": user_query})
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    system_response = completion["choices"][0]["message"]["content"]

    spinner.succeed("Assistant Response Complete")

    print("\n")
    print(system_response)

    system_message = {"role": "assistant", "content": system_response}
    messages.append(system_message)

    return messages, system_response


def chat_transcript(messages):
    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript


def gen_ai_sound(response, voice_id, output_file):

    choice = (list(name_id_dict.values()))[voice_id]

    json_data = {
        "text": response,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
        },
    }

    spinner.start("Converting to Audio")

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{choice}",
        headers=headers,
        json=json_data,
    )

    spinner.succeed("Audio Converted")
    
    print(response.status_code)

    content = response.content
    with open(output_file, "wb") as file:
        file.write(content)

    


def run_chat(num_responses, recording_duration, choice):
    init_prompt = "You are to act as a helpful assistant. The assistant is helpful, creative, clever, and very friendly."
    messages = [{"role": "system", "content": init_prompt}]
    for x in range(0, num_responses):

        record_audio("output.wav", duration=recording_duration)

        chat = transcribe("output.wav")

        messages, just_text = get_assistant_response(messages, chat)

        transcript = chat_transcript(messages)

        print(transcript)

        gen_ai_sound(just_text, choice)
        
    return transcript


