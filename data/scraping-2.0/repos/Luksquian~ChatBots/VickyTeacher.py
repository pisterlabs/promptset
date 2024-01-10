# *****************************************************************************************************
# *****************************************************************************************************
# *************************** CODIGO CREADO Y COMPARTIDO POR LKTECHAI *********************************
# *****************************************************************************************************
# *****************************************************************************************************

import speech_recognition as sr
import openai
from gtts import gTTS
from pygame import mixer
import os
import time as ti
import random
from tkinter import *   #Libreria para manejo de interfaces graficas

openai.api_key = "<KEY-OPENAI>"

# Iniciar TKinter
aplicacion = Tk()

# Definir tamaño de ventana: Resolucion+posx+posy
aplicacion.geometry('480x650+300+20')

# Evitar maximizacion
aplicacion.resizable(0,0)

# Definimos Titulo de ventana
aplicacion.title("Vicky")

# Definimos color de fondo de ventana
aplicacion.config(bg='gray')

# Definimos el fondo y background de la interfaz grafica
fondo = PhotoImage(file="imagenes//vicky.png")
background = Label(image=fondo, text="Imagen de fondo")
background.place(x=0, y=-68, relwidth=1, relheight=1)

# Definimos el cuadro de texto donde mostraremos el texto de pregunta y respuesta
text_box = Text(aplicacion, height=7, width=60)
text_box.insert('end', "Press Speak button!")
text_box.pack(pady=20)
text_box.place(x=0, y=500)


# definimos cómo vamos a tomar el audio del mic y pasarlo a texto
def transformar_audio_a_texto():

    r = sr.Recognizer()

    with sr.Microphone() as origen:
        r.pause_threshold = 0.5
        print("Let Go!")
        audio = r.listen(origen)
        try:
            pedido = r.recognize_google(audio, language="en-GB")
            print("You: " + pedido)
            return pedido

        except sr.UnknownValueError:
            print("Ups, No entendi!")
            return "Ups, No entendi!"

        except sr.RequestError:
            print("Ups, no hay servicio!")
            return "Ups, no hay servicio!"

        except:
            print("Ups, algo salio mal!")
            return "Ups, algo salio mal!"

			
# Definimos cómo dado un texto (mensaje) lo pasamos a audio 
def hablar_en(mensaje):
    volume = 0.7
    tts = gTTS(mensaje, lang="en", slow=False)
    ran = random.randint(0, 9999)
    filename = 'TempEn' + format(ran) + '.mp3'
    tts.save(filename)
    mixer.init()
    mixer.music.load(filename)
    mixer.music.set_volume(volume)
    mixer.music.play()

    while mixer.music.get_busy():
        ti.sleep(0.3)

    mixer.quit()
    os.remove(filename)


# definimos la comunicacion con OpenAI
# Conversation es el preseteo que le damos a chatgpt para que tome el rol que queremos
def traer_respuesta(question):

    conversation = "Vicky es un chatbot en el rol de una profesora de ingles amable y divertida." \
					"Ella propone temas, preguntas en ingles y espera que el interlocutor responda en ingles tambien. " \
					"Luego, si es que lo amerita, ella corrige la gramatica de la respuesta del interlocutor proponiendo una alternativa correcta " \
					"con frases como 'you can tell him better this way...' or 'tray with ...'. " \
					"El objetivo principal es que el interlocutor se divierta y aprenda ingles."

    conversation += "\nYou: " + question + "\nVicky:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=conversation,
        temperature=0.5,
        max_tokens=60,
        top_p=0.3,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        stop=["\n", " You:", " Vicky:"]
    )
    answer = response.choices[0].text.strip()
    return answer


# Definimos la funcion que se ejecutará cuando demos clic en el boton de hablar
def talk():
    question = transformar_audio_a_texto().lower()
    text_box.insert('end', 'You: ' + question + '\n')
    answer = traer_respuesta(question)
    text_box.insert('end', 'Vicky: ' + answer + '\n')
    print("Vicky: " + answer + "\n")
    hablar_en(answer)


# Definimos el boton que iniciara el speak entre el interlocutor y chatgpt
my_button = Button(aplicacion, text="Speak", command=talk, width=30)
my_button.pack(pady=20)
my_button.place(x=120, y=620)

# Evitar que la pantalla se cierre
aplicacion.mainloop()

