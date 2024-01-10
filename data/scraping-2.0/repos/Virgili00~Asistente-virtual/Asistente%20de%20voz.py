import pyaudio as pa
import os
import openai 
import pyttsx3
import webbrowser
import funciones 
texto = ''
#inicializo el motor de voz-----------
engine = pyttsx3.init()
engine.setProperty('rate',150)
voices = engine.getProperty("voices")
engine.setProperty("voice",voices[0].id)
#-------------------------------------
#saludo al usuario
engine.say("Hola Buen dia, Federico ")
engine.runAndWait()
#-------------------------------------
#ciclo del programa
while ('CHAO' or 'CHAU') not in texto:
    engine.say("Te escucho")
    engine.runAndWait()
    print("Te escucho")
    texto=funciones.talk()
    texto=texto.upper() #pongo todo el texto en  mayusculas para que no haya problemas
    print(texto)
    
    if 'YOUTUBE' in texto: #si en la frase que deci esta YOUTUBE te abre youtube
        engine.say("abriendo youtube")
        engine.runAndWait()
        print("abriendo youtube")
        webbrowser.open("https://www.youtube.com/") 
        
    elif 'BLOC DE NOTAS' in texto: #si en la frase que deci esta BLOC DE NOTAS entra a un archivo para escribir texto
        funciones.bloc()

    elif 'HABLEMOS' in texto:#si en la frase que deci esta HABLEMOS se conecta con ChatGPT
        while 'LISTO' not in texto: #listo es la palabra clave para salir del chat
            engine.say("okay hablemos")
            engine.runAndWait()
            print("okay que hablamos?")
            funciones.hablemos()
            engine.say("¿Algo mas?")
            engine.runAndWait()
            texto=funciones.talk() #aca te da la posibilidad de decir Listo para poder salir del ciclo
            texto=texto.upper()



engine.say("adios")
engine.runAndWait()            

