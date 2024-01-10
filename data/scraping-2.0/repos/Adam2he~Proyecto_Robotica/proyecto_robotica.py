#!/usr/bin/python

# Importamos la libreria de PySerial
import serial
import pyttsx3 as voz
import speech_recognition as sr
import subprocess as sub
from datetime import datetime
import openai

#Creamos la IA
openai.api_key="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"
completion=openai.Completion.create(engine="curie",prompt="¿Que es chatgpt?",max_tokens="2048")
print(completion.choices[0].text)

# Abrimos el puerto de la placa a 9600
PuertoSerie = serial.Serial('COM6', 9600)
#Creamos la voz
voice = voz.init()
voices = voice.getProperty('voices')
voice.setProperty('voice', voices[0].id)
voice.setProperty('rate', 140)

def say(text):
  voice.say(text)
  voice.runAndWait()

while True:

  recognizer=sr.Recognizer()
  #Activar microfono
  with sr.Microphone() as source:
    print("Escuchando...")
    audio = recognizer.listen(source, phrase_time_limit=3)

  try:
    comando=recognizer.recognize_google(audio,language='es-ES')
    print("Creo que dijiste: "+comando)

    comando=comando.lower()
    comando=comando.split(' ')

    if 'temperatura' in comando:
      sArduino = PuertoSerie.readline()

      print("Me ha llegado " + str(sArduino))
      i = 10 * (int(sArduino[0]) - 48) + (int(sArduino[1]) - 48)  # De ASCII a entero
      print(i)
      say("La temperatura es de "+str(i)+"grados")
    else:
      say("No te he entendido")

  except:
    say("No te he entendido")

