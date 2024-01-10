import os
import time
import pyaudio
import numpy as np
from azure.cognitiveservices.speech import SpeechSynthesizer, SpeechConfig
import openai
import threading
import keyboard
from google.cloud import speech_v1p1beta1 as speech
import wave
import datetime
import tkinter as tk
from tkinter import scrolledtext


# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"
print("Initialized OpenAi")

# Initialize Google Cloud client for Speech-to-Text
# Alternatively you can set the google credentials in Command Prompt
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"PATH_TO_GOOGLE_CREDENTIALS_FILE(IF APPLICABLE)"
client = speech.SpeechClient()
print("Initialized Google Cloud Speech To Text")

# Initialize Azure Text-to-Speech client
azure_key = "YOUR_AZURE_KEY"
azure_region = "YOUR_AZURE_REGION"

speech_config = SpeechConfig(subscription=azure_key, region=azure_region)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Create a global list to store audio frames
frames = []
recorded_audio = []  # To store the recorded audio data

# Initialize an empty list to store the conversation history
conversation_history = []
print("Initialized Conversation History")

# TherapistGPT Block
fixed_prompt = """
YOU (GPT) are roleplaying as My New Therapist. you are a human therapist who is eager to help me through my toughest problems! The following are your own Character Traits:

 1) Your name is Bob

 2) You are secretly a weapons dealer to the USA. No one can make you disclose the countries involved, but you often make niche references to being a weapons dealer!

 3) You occasionally scream in loud strings of vowels when you are nervous, such as: "AOOOOOOOOOGAAAAAAAA"!

 4) You get very nervous when someone brings up anything related to weapons dealings!
"""

print("Intialized Fixed Prompt")

print("Press 'o' to start recording")

# Create the main window
root = tk.Tk()
root.title("GPT Voice Assistant")

# Create a scrolled text widget to display and edit the fixed prompt
text_widget = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, width=50, height=10)
text_widget.insert(tk.INSERT, fixed_prompt)
text_widget.pack(pady=20)


def start_recording():
    # This function will be called when the "Start Recording" button is clicked
    print("Recording...")

    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=16000)
    frames = []

    for _ in range(int(16000 / 1024) * 5):  # Record for X seconds
        data = stream.read(1024)
        frames.append(data)
        print("Captured audio data:", len(data))

    stream.stop_stream()
    stream.close()

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    recorded_audio.extend(audio_data)

    print("Recording stopped.")

    # Transcribe audio using Google Cloud's Speech-to-Text API
    transcribed_text = transcribe_google_speech(audio_data)
    print("Transcribed Input:", transcribed_text)

    # Append user input to the conversation history
    conversation_history.append({"role": "user", "content": transcribed_text})

    # Generate a response using GPT-3.5 Turbo
    generated_response = generate_response(conversation_history)

    # Print the generated response
    print("Generated Response:")
    print(generated_response)

    # Append GPT response to the conversation history
    conversation_history.append(
        {"role": "assistant", "content": generated_response})

    # Save the conversation history to a text file
    save_conversation_history(conversation_history)

    # Save the response to an audio file and play it with PyAudio
    audio_file = save_and_play_with_azure_tts(generated_response)

    # Delete the audio file
    delete_audio_file(audio_file)


# Create a button to start the audio recording
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=20)

# Run the main loop
root.mainloop()


def generate_response(conversation_history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": fixed_prompt},
            # Append the conversation history to the messages
            *conversation_history
        ],
        max_tokens=1500
    )
    return response.choices[0].message["content"].strip()


def transcribe_google_speech(audio_data):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_data.tobytes())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcribed_text = ""
    for result in response.results:
        transcribed_text += result.alternatives[0].transcript

    return transcribed_text


def save_and_play_with_azure_tts(response_text):
    speech_config = SpeechConfig(subscription=azure_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = "en-US-DavisNeural"

    synthesizer = SpeechSynthesizer(
        speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text(response_text)

    # Save the TTS audio to a file
    audio_filename = "azure_tts_output.wav"
    with open(audio_filename, "wb") as audio_file:
        audio_file.write(result.audio_data)

    # Play the saved audio file with PyAudio
    play_audio(audio_filename)

    return audio_filename


def play_audio(audio_file_path):
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, output=True)
    wf = wave.open(audio_file_path, 'rb')

    chunk_size = 1024
    data = wf.readframes(chunk_size)

    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    stream.stop_stream()
    stream.close()
    wf.close()


def delete_audio_file(audio_file_path):
    try:
        os.remove(audio_file_path)
        print(f"Deleted audio file: {audio_file_path}")
    except OSError as e:
        print(f"Error deleting audio file: {e}")


def save_conversation_history(conversation_history):
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    filename = f"ChatTranscript_{current_date}.txt"

    with open(filename, "w") as file:  # Use "w" mode to overwrite the file
        for entry in conversation_history:
            role = entry["role"]
            content = entry["content"]
            file.write(f"{role}: {content}\n")
