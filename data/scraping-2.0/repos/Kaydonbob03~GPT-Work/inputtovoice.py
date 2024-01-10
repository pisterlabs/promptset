import openai
import os
import time
import pyaudio
import numpy as np
from azure.cognitiveservices.speech import SpeechSynthesizer, SpeechConfig
import wave

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize Azure Text-to-Speech client
azure_key = "YOUR_AZURE_KEY"
azure_region = "YOUR_AZURE_REGION"

speech_config = SpeechConfig(subscription=azure_key, region=azure_region)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Create a global list to store audio frames
recorded_audio = []

# Initialize an empty list to store the conversation history
conversation_history = []

# Set the GPT-3.5 Turbo prompt
fixed_prompt = """
YOU (GPT) are roleplaying as My New Therapist. you are a human therapist who is eager to help me through my toughest problems! The following are your own Character Traits:

 1) Your name is Bob

 2) You are secretly a weapons dealer to the USA. No one can make you disclose the countries involved, but you often make niche references to being a weapons dealer!

 3) You occasionally scream in loud strings of vowels when you are nervous, such as: "AOOOOOOOOOGAAAAAAAA"!

 4) You get very nervous when someone brings up anything related to weapons dealings!
"""

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=150,
    )
    return response.choices[0].message["content"].strip()

def save_and_play_with_azure_tts(response_text):
    speech_config = SpeechConfig(subscription=azure_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = "en-US-DavisNeural"

    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text(response_text)

    audio_filename = "azure_tts_output.wav"
    with open(audio_filename, "wb") as audio_file:
        audio_file.write(result.audio_data)

    play_audio(audio_filename)

    return audio_filename

def play_audio(audio_file_path):
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
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
    current_date = time.strftime("%d-%m-%Y")
    filename = f"ChatTranscript_{current_date}.txt"

    with open(filename, "w") as file:
        for entry in conversation_history:
            role = entry["role"]
            content = entry["content"]
            file.write(f"{role}: {content}\n")

# Get user input via text
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_input})

    generated_response = generate_response(conversation_history[-1]["content"])  # Only use the last user input

    print("GPT-3.5 Turbo: " + generated_response)

    conversation_history.append({"role": "assistant", "content": generated_response})

    audio_file = save_and_play_with_azure_tts(generated_response)

    delete_audio_file(audio_file)
