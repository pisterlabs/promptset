import openai
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import time 
from gtts import gTTS
import os

# Set OPENAI API key
# openai.api_key = ""

def transcribe_audio_to_text(audio_data, fs):
    recognizer = sr.Recognizer()
    audio = sr.AudioData(audio_data.tobytes(), fs, 2)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e: 
        print("Se encontró un error: ", e)

def generate_response(prompt): 
    response = openai.Completion.create(
        engine= "text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]

def speak_text(text):
    tts = gTTS(text=text, lang='en')
    filename = "temp.mp3"
    tts.save(filename)
    os.system("mpg321 " + filename)

def main():
    fs = 16000
    while True:
        print("Di `Hola` para activar nuestro servicio de chat")
        print("Dime qué necesitas")
        recording = sd.rec(int(fs * 3), samplerate=fs, channels=1)
        sd.wait()  # Espera hasta que termine la grabación
        recording = np.squeeze(recording)  # Remueve una dimensión innecesaria

        try: 
            transcription = transcribe_audio_to_text(audio)
            if transcription is not None:    
                if transcription.lower() == "hola":
                    print(f"Dijiste: {transcription}")

                    # Generar una respuesta usando GPT-3
                    response = generate_response(transcription)
                    print(f"Te decimos: {response}")

                    # Leamos la respuesta usando text-to-speech
                    speak_text(response)
                else:
                    print("No entendí eso. Por favor, di `Hola` para activar el servicio de chat.")
        except Exception as e:
                print(f"Ups! ha ocurrido un error: {e}")

if __name__     == "__main__":
    main()  
