import os
import openai
import json
import pyaudio
import wave
import pyttsx3
import ApiKey as AK


openai.api_key = AK.API

# Note: you need to be using OpenAI Python v0.27.0 for the code below to work

def audioRecog(filepath):
    
    audio_file= open(filepath, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    voiceContent = transcript["text"]
    return voiceContent


filepath = "AUDIOCHAT/AudioChat.wav"

if __name__ == "__main__":
    text = audioRecog(filepath)
    print(text)
