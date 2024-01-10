import os

import gtts
from playsound import playsound
from config import openai_key
from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
from time import sleep

client = OpenAI(api_key=openai_key)


def playSpeech(text, language):
    # make request to google to get synthesis
    #tts = gtts.gTTS("Hello world")

    # save the audio file
    #tts.save("hello.mp3")

    # play the audio file
    #playsound("hello.mp3")
    
    # in spanish
    print(language)
    tts = gtts.gTTS(text, lang=language)
    tts.save("sound.mp3")
    playsound("sound.mp3")

def recordSpeech(time):
    fs = 44100  # Sample rate
    seconds = time  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Recording...")
    sd.wait()  # Wait until recording is finished
    write('speech.wav', fs, myrecording)  # Save as WAV file
    print("Finished recording.")

    audio_file = open("speech.wav", "rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
    )
    return transcript.text
