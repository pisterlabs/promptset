import tkinter as tk
from tkinter import Entry, Button, Label
import pygame
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import os
import speech_recognition as sr
from termcolor import colored
import openai
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, AudioFileClip
import pyttsx3

# Cargar variables de entorno desde el archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Definir la plantilla para la generación de respuestas
template = """
Siri es un gran modelo lingüístico entrenado por OpenAI.
Siri tiene una personalidad amigable y es muy útil, le encanta ayudar a los usuarios a realizar tareas en su computadora.
Siri está diseñado para ayudar en una amplia gama de tareas, desde responder a preguntas sencillas hasta proporcionar explicaciones detalladas y debates sobre una gran variedad de temas. Como modelo lingüístico, Assistant es capaz de generar textos similares a los humanos a partir de la información que recibe, lo que le permite entablar conversaciones naturales y ofrecer respuestas coherentes y relevantes para el tema en cuestión.
En general, Siri es una potente herramienta que puede ayudarte con una gran variedad de tareas y proporcionarte valiosos conocimientos e información sobre una amplia gama de temas. Tanto si necesitas ayuda con una pregunta concreta como si sólo quieres mantener una conversación sobre un tema en particular, Siri está aquí para ayudarte.
{history}
Human: {input_text}
Siri:
"""
prompt = PromptTemplate(input_variables=["input_text", "history"], template=template)

# Definir la memoria de conversación
memory = ConversationBufferMemory(memory_key="chat_history")

# Crear una instancia de OpenAI
llm = OpenAI(temperature=0.5)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Definir función de búsqueda personalizada
def search(query):
    # Aquí puedes implementar tu propia lógica de búsqueda
    return "Resultados de búsqueda para: " + query

# Definir lista de herramientas
tools = [
    Tool(
        name="Current Search",
        func=search,
        description="útil para responder a preguntas sobre la actualidad o el estado actual del mundo"
    ),
]

# Inicializar agente con herramientas y configuración
zero_shot_agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
    memory=memory,
    verbose=True,
    prompt=prompt
)

# Función para procesar la entrada del usuario
def process_input(input_text):
    if input_text.lower() == "salir":
        return None
    else:
        response = generate_response(input_text)
        return response

# Función para generar respuesta utilizando el agente
def generate_response(input_text):
    response = zero_shot_agent(input_text)
    respuesta = response['output']
    return respuesta

# Función para leer la entrada del usuario desde el teclado
def read_input():
    while True:
        input_text = input(colored("Escribe algo: ", "green"))
        response = process_input(input_text)
        if response == False:
            break
        else:
            print(colored("Siri:", "yellow"), response)

# Función para generar respuesta utilizando el agente
def generate_response(input_text):
    response = zero_shot_agent(input_text)
    respuesta = response['output']
    return respuesta

# Función para leer la entrada del usuario desde el teclado
def read_input():
    while True:
        input_text = input(colored("Escribe algo: ", "green"))
        response = process_input(input_text)
        if response == False:
            break
        else:
            print(colored("Siri:", "yellow"), response)

# Función para generar la respuesta y crear un video con ella
def process_audio_video(frase):
    # Configurar el motor de síntesis de voz
    engine = pyttsx3.init()

    # Seleccionar una voz (puedes ajustar esto)
    selected_voice_id = "spanish-latin-am"
    engine.setProperty('voice', selected_voice_id)

    # Generar el archivo de audio con gTTS
    tts = gTTS(text=frase, lang='es')
    tts.save("audio.mp3")

    # Rutas de los archivos de video y audio
    audio_path = './audio.mp3'
    video_path = './girl.mp4'

    # Cargar el video y el audio
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Ajustar la duración del video para que coincida con la duración del audio
    video = video.set_duration(audio.duration)

    # Combinar el audio con el video
    video = video.set_audio(audio)

    # Guardar el nuevo video con el audio combinado y la duración ajustada
    output_path = 'video_con_audio.mp4'
    video.write_videofile(output_path, codec='libx264')

def send_text():
    user_input = input_entry.get()
    response = process_input(user_input)
    response_label.config(text=response)
    process_audio_video(response)

def play_video(video_path):
    pygame.mixer.music.load(video_path)
    pygame.mixer.music.play()

def create_interface():
    root = tk.Tk()
    root.title("Interfaz de Siri")
    root.geometry("800x600")

    input_entry = Entry(root, width=50)
    input_entry.pack(pady=20)

    send_button = Button(root, text="Enviar", command=send_text)
    send_button.pack()

    response_label = Label(root, text="", wraplength=600)
    response_label.pack(pady=20)

    root.mainloop()

def process_input(input_text):
    if input_text.lower() == "salir":
        return None
    else:
        response = generate_response(input_text)
        return response

def generate_response(input_text):
    response = zero_shot_agent(input_text)
    respuesta = response['output']
    return respuesta

def process_audio_video(frase):
    engine = pyttsx3.init()
    selected_voice_id = "spanish-latin-am"
    engine.setProperty('voice', selected_voice_id)
    tts = gTTS(text=frase, lang='es')
    tts.save("audio.mp3")
    audio_path = './audio.mp3'
    video_path = './girl.mp4'
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    video = video.set_duration(audio.duration)
    video = video.set_audio(audio)
    output_path = 'video_con_audio.mp4'
    video.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    initialize_agent()

    pygame.init()
    pygame.mixer.init()

    print(colored("¡Buenas! Soy Siri, ¿En qué puedo ayudarte?", "magenta"))
    create_interface()