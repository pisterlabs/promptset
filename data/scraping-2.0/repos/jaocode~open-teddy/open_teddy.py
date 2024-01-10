from operator import truediv
import tts.tts_factory as tts_factory
from ai.openai_chatgpt import OpenAIChatGPT as AI_Engine
import os.path

import sounddevice as sd
import soundfile as sf

def speak(text):
    print ("Synthesizing speech.")
    audio_file = TTS.create_speech_audio (text)
    
    print ("Speaking.")
    data, fs = sf.read(audio_file)
    sd.play(data, fs, blocking=True)

def basic_response_processing(query, TTS, AI):
    resp = AI.query_ai (query)
    print ("Response: " + resp)
    speak (resp)
    

if __name__ == '__main__':
    prompt = "The following is a conversation with Teddy Ruxpin. He is funny, creative, clever, and sarcastic at times. Sometimes he will tell stories about his old friends. He likes meeting people and making new friends."
    
    # See tts_factory.py file for available options. 
    TTS = tts_factory.use_elevenlabs()
    AI = AI_Engine(prompt=prompt)

    go = True
    while go:
        print ("I'm ready for your next query.\n")
        query = input()

        if query == "!exit":
            go = False
            continue
        
        print ("Processing response.\n")
        basic_response_processing(query, TTS, AI)
        print ("done")