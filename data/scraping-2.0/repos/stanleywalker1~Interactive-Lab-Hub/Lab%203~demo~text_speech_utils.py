# text_speech_utils.py

import openai
import sounddevice as sd
import audiofile as af
from scipy.io.wavfile import write
from gtts import gTTS

import multiprocessing
import pyttsx3
import keyboard

def say(text):
    audio_filename = "temp_speech_output.mp3"
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save(audio_filename)
    play_audio(audio_filename)

def record_audio(filename, sec, sr = 44100):
    audio = sd.rec(int(sec * sr), samplerate=sr, channels=1, blocking=False)
    sd.wait()
    write(filename, sr, audio)

def record_audio_manual(filename, sr = 44100):
    input("  ** Press enter to start recording **")
    audio = sd.rec(int(10 * sr), samplerate=sr, channels=1)
    input("  ** Press enter to stop recording **")
    sd.stop()
    write(filename, sr, audio)

def play_audio(filename):
    signal, sr = af.read(filename)
    sd.play(signal, sr)

def transcribe_audio(filename):
    audio_file= open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    return transcript

def translate_audio(filename):
    audio_file= open(filename, "rb")
    translation = openai.Audio.translate("whisper-1", audio_file)
    audio_file.close()
    return translation

def save_text_as_audio(text, audio_filename):
    myobj = gTTS(text=text, lang='en', slow=False)  
    myobj.save(audio_filename)
