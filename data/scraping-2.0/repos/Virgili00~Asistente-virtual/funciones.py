import speech_recognition as sr
import pyaudio as pa
import pyttsx3
import openai
import os
import API
api=API.Api_key #aca se ingresa tu API_KEY de OpenAi para poder utilizarlo
openai.api_key = api
texto = '' 
#inicializo el motor de voz de forma general porq lo usare modificado en varias funciones
engine = pyttsx3.init()
engine.setProperty('rate',150)
#en esta funcion es cuando quiero pasar de texto a voz algo puntual.
def texttospeech(texto):
            engine.say(texto)
            engine.runAndWait()
    

def hablemos():
    escribir= talk()
    if 'nada' not in escribir: #solo  te va a dar una respuesta chatgpt si en tu oracion no decis "nada"
        print(escribir)
        #se inicializa chatgpt
        resource = openai.Completion.create(
                                             engine="text-davinci-003", 
                                            prompt=escribir, 
                                             max_tokens=150)
            
        respuesta=resource.choices[0].text
        texttospeech(respuesta)
        print(respuesta)
    else:
        engine.say("okay bueno")
        engine.runAndWait()
        
    
def bloc():
        print("Abriendo bloc de notas")
        engine.say("Abriendo bloc de notas")
        engine.runAndWait()
        engine.say("¿que desea escribir?")
        engine.runAndWait()
        
        print("que desea escribir?")
        #lee el audio
        escribir = talk()
        #traduce a texto el audio
        engine.say("dijiste"+escribir)
        engine.runAndWait()
        engine.say("¿quieres guardar lo que acabas de anotar?")
        engine.runAndWait()    
        confirmo = talk() 
        confirmo = confirmo.upper()
        if 'QUIERO' in confirmo: #aca guarda el archivo. si "QUIERO" esta en la confirmacion
            archivo = open("pensamiento.txt","a") #abre o crea el archivo
            archivo.write('\n') #introduce un salto de linea siempre que escribamos el archivo  no escriba todo junto
            archivo.write(escribir)#escribe el archivo con el texto que le dictamos
            archivo.close()
            engine.say("guardado")
            engine.runAndWait()
 
             
def talk():
    #---inicializo el reconocedor de voz
    reco = sr.Recognizer()
    reco.energy_threshold = 3000
    reco.pause_threshold = 1
    reco.non_speaking_duration = 0.8
    mic = sr.Microphone()
    #-------------
    #intenta reconocer el texto si se detecta un error printea que hubo un error
    try:
        with mic as source:
            audio = reco.listen(source)
            texto = reco.recognize_google(audio, 
                                            language="es-MX")
            return texto
    except:
        print("hubo un error")