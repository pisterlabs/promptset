# Import the necessary libraries
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import numpy as np
import openai
import os
import requests
import re
from colorama import Fore, Style, init
import datetime
import base64
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import io
import threading

# Initialize colorama and load environment variables
init()
load_dotenv()


# Define a function to open and read a file
def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        return infile.read()


# Get the API keys from the environment variables
api_key = os.getenv("openaiapikey2")
elapikey = os.getenv("elabapikey")

# Initialize the conversation and chatbot variables
conversation1 = []
chatbot1 = open_file("chatbot1.txt")


# Define the function to communicate with the OpenAI GPT-3 model
def chatgpt(
    api_key,
    conversation,
    chatbot,
    user_input,
    temperature=0.9,
    frequency_penalty=0.2,
    presence_penalty=0,
):
    # Set the API key and prepare the conversation messages
    openai.api_key = api_key
    conversation.append({"role": "user", "content": user_input})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])

    # Make the request to the GPT-3 model
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input,
    )

    # Get the chat response and add it to the conversation
    chat_response = completion["choices"][0]["message"]["content"]
    conversation.append({"role": "assistant", "content": chat_response})

    return chat_response


# Define the function to convert text to speech
def text_to_speech(
    text, voice_id, api_key, output_file="output.mp3", append_audio=True
):
    # Prepare the request to the Eleven Labs API
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.6, "similarity_boost": 0.85},
    }

    # Make the request
    response = requests.post(url, headers=headers, json=data)

    # If the request was successful, process and play the audio
    if response.status_code == 200:
        new_audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")

        if append_audio:
            try:
                existing_audio = AudioSegment.from_mp3(output_file)
                combined_audio = existing_audio + new_audio
                combined_audio.export(output_file, format="mp3")
            except FileNotFoundError:
                new_audio.export(output_file, format="mp3")

        play(new_audio)
    else:
        print("Error:", response.text)


# Define the function to print colored text to the console
def print_colored(agent, text):
    agent_colors = {
        "Brillua:": Fore.YELLOW,
    }
    color = agent_colors.get(agent, "")
    print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")


# Specify the voice ID for the text to speech conversion
voice_id1 = "AcoYNi79OOiT50tw0ub0"


# Define the function to record audio and transcribe it to text
def record_and_transcribe(duration=60, fs=44100):
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    print("Recording complete.")
    filename = "myrecording.wav"
    sf.write(filename, myrecording, fs)
    with open(filename, "rb") as file:
        openai.api_key = api_key
        result = openai.Audio.transcribe("whisper-1", file)
    transcription = result["text"]
    return transcription


# Create a new Tkinter window
window = tk.Tk()

# Define a variable to hold the audio data
recording = np.zeros((44100 * 8,))


# Define a function to start the recording and process the audio
def start_recording():
    global recording
    user_message = record_and_transcribe()
    response = chatgpt(api_key, conversation1, chatbot1, user_message)
    print_colored("Brillua:", f"{response}\n\n")
    user_message_without_generate_image = re.sub(
        r"(Response:|Narration:|Image: generate_image:.*|)", "", response
    ).strip()
    text_to_speech(user_message_without_generate_image, voice_id1, elapikey)


# Define a function to stop the recording
def stop_recording():
    sd.stop()


# Create a button to start the recording
start_button = tk.Button(
    window,
    text="Start Recording",
    command=lambda: threading.Thread(target=start_recording).start(),
)

# Create a button to stop the recording
stop_button = tk.Button(window, text="Stop Recording", command=stop_recording)

# Create a text area to show the ongoing dialogue
text_area = tk.Text(window)

# Create a button to end the session
end_button = tk.Button(window, text="End Session", command=window.quit)

# Add the buttons and text area to the window
start_button.pack()
stop_button.pack()
text_area.pack()
end_button.pack()

# Start the Tkinter event loop
window.mainloop()


# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# import openai
# import os
# import requests
# import re
# from colorama import Fore, Style, init
# import datetime
# import base64
# from pydub import AudioSegment
# from pydub.playback import play
# from dotenv import load_dotenv
# import io

# init()
# load_dotenv()


# def open_file(filepath):
#     with open(filepath, "r", encoding="utf-8") as infile:
#         return infile.read()


# api_key = os.getenv("openaiapikey2")
# elapikey = os.getenv("elabapikey")

# conversation1 = []
# chatbot1 = open_file("chatbot1.txt")


# def chatgpt(
#     api_key,
#     conversation,
#     chatbot,
#     user_input,
#     temperature=0.9,
#     frequency_penalty=0.2,
#     presence_penalty=0,
# ):
#     openai.api_key = api_key
#     conversation.append({"role": "user", "content": user_input})
#     messages_input = conversation.copy()
#     prompt = [{"role": "system", "content": chatbot}]
#     messages_input.insert(0, prompt[0])
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-0613",
#         temperature=temperature,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty,
#         messages=messages_input,
#     )
#     chat_response = completion["choices"][0]["message"]["content"]
#     conversation.append({"role": "assistant", "content": chat_response})
#     return chat_response


# def text_to_speech(
#     text, voice_id, api_key, output_file="output.mp3", append_audio=True
# ):
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
#     headers = {
#         "Accept": "audio/mpeg",
#         "xi-api-key": api_key,
#         "Content-Type": "application/json",
#     }
#     data = {
#         "text": text,
#         "model_id": "eleven_monolingual_v1",
#         "voice_settings": {"stability": 0.6, "similarity_boost": 0.85},
#     }
#     response = requests.post(url, headers=headers, json=data)

#     if response.status_code == 200:
#         new_audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")

#         if append_audio:
#             try:
#                 existing_audio = AudioSegment.from_mp3(output_file)
#                 combined_audio = existing_audio + new_audio
#                 combined_audio.export(output_file, format="mp3")
#             except FileNotFoundError:
#                 new_audio.export(output_file, format="mp3")

#         play(new_audio)
#     else:
#         print("Error:", response.text)


# def print_colored(agent, text):
#     agent_colors = {
#         "Brillua:": Fore.YELLOW,
#     }
#     color = agent_colors.get(agent, "")
#     print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")


# voice_id1 = "AcoYNi79OOiT50tw0ub0"


# def record_and_transcribe(duration=8, fs=44100):
#     print("Recording...")
#     myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
#     sd.wait()
#     print("Recording complete.")
#     filename = "myrecording.wav"
#     sf.write(filename, myrecording, fs)
#     with open(filename, "rb") as file:
#         openai.api_key = api_key
#         result = openai.Audio.transcribe("whisper-1", file)
#     transcription = result["text"]
#     return transcription


# while True:
#     user_message = record_and_transcribe()
#     response = chatgpt(api_key, conversation1, chatbot1, user_message)
#     print_colored("Brillua:", f"{response}\n\n")
#     user_message_without_generate_image = re.sub(
#         r"(Response:|Narration:|Image: generate_image:.*|)", "", response
#     ).strip()
#     text_to_speech(user_message_without_generate_image, voice_id1, elapikey)
