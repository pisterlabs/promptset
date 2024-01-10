import pyaudio
import wave
import keyboard

import openai
from langchain.llms import OpenAI

import pyttsx3

def transcribe_audio(output_file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    
    print("Starting recording... Press Enter when done.")

    # Continuously read audio data until Enter key is pressed
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        if keyboard.is_pressed('enter'):
            break

    # Wrapped in an "input" to log the pressed Enter key and not use it below
    input("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(output_file, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
    
    # open the audio file
    audio_file = open(output_file, "rb")

    # get a transcript to be used as a prompt
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
    return transcript['text']
    
    
if __name__ == "__main__":
    
    # initialize the engine that reads the LLM's answers
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    
    while True:
    
        prompt = transcribe_audio('recording.wav')
        
        # Call the llm to reply to the prompt
        llm = OpenAI(temperature=0.1)
        resp = llm(prompt).replace("\n","")

        # Read the output aloud
        engine.say(resp)
        engine.runAndWait()
        
        engine.say("Do you wish to ask anything else? If so, say yes.")
        engine.runAndWait()
        
        reply = transcribe_audio('recording.wav')
        reply = ''.join(filter(str.isalpha, reply.lower()))
        
        if reply != "yes":
            break
