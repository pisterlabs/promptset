from pydub import AudioSegment
from pydub.playback import play
from pygame import mixer
import time
from pathlib import Path
from openai import OpenAI
import os
import sounddevice as sd
import numpy as np
import soundfile as sf

VOICE_FOLDER = "voices" 
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_VOICE = "onyx"

def create_gpt_client():
    with open('E:\\dev\\api_keys\\OPENAI_API_KEY', 'r') as file:
        api_key = file.read().strip()
    return  OpenAI(api_key=api_key)


def create_speech_from_text(text, model="tts-1", voice=DEFAULT_VOICE, name="agent_voice"):
    
    print("Creating speech from text...")
    file_name = f"{name}.mp3"
    local_dir_path = VOICE_FOLDER
    local_file_path = os.path.join(local_dir_path, file_name)
    if not os.path.exists(local_dir_path): os.mkdir(local_dir_path)
    

    client = create_gpt_client()
    # Generate speech from text
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    
    response.stream_to_file(local_file_path)
    # Save the speech to the specified file path
    
    return local_file_path

def play_audio(file_path, stop_flag=None):
    mixer.init()
    mixer.music.load(file_path)
    mixer.music.play()

    if stop_flag is None:
        # If stop_flag is not provided, wait for the audio to finish playing
        while mixer.music.get_busy():
            time.sleep(0.1)
    else:
        # If stop_flag is provided, wait for the flag to be set
        while not stop_flag.is_set():
            time.sleep(0.1)

def list_audio_devices():
    print("Available audio devices:")
    for i, device_info in enumerate(sd.query_devices()):
        print(f"{i}: {device_info['name']}")

def record_audio(name="user_input", duration=5, sample_rate=44100, input_device=None):
    file_name = f"{name}.wav"
    local_dir_path = VOICE_FOLDER
    local_file_path = os.path.join(local_dir_path, file_name)
    if not os.path.exists(local_dir_path): os.mkdir(local_dir_path)
    if input_device is None:
        input_device = sd.default.device["input"]
    print("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16, device=input_device)
    sd.wait()
    sf.write(local_file_path, audio_data, sample_rate)
    
    
    
    print(f"Recording saved to {local_file_path}")
    return local_file_path

def voice_to_text(file_path, model="whisper-1", name="user_input"):
    print(f"Converting voice to text {file_path}...")
    audio_file= open(file_path, "rb")
    client = create_gpt_client()
    transcript = client.audio.transcriptions.create(
    model=model, 
    file=audio_file,
    response_format="text"
    )

    return transcript
    

# Get text from chatbot
# text_to_speak = "Today is a wonderful day to build something people love!"
# agent_voice_file = create_speech_from_text(text_to_speak)
# agent_voice_file = "voices\\agent_voice.mp3"
# play_audio(agent_voice_file)

# Get voice from user
# user_voice_file_path = record_audio("user_input", duration=5)
# user_text = voice_to_text(user_voice_file_path)
# print(user_text)





