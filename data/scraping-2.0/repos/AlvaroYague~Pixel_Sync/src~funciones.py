# Funciones a exportar
import pandas as pd
from googletrans import Translator
import pyttsx3
import streamlit as st
import openai
import speech_recognition as sr

# Funciones para limpieza de datos

def apply_genre_mapping(genre):
    return genre_mapping.get(genre)       # Función para mapear géneros


def categorize(column):                   # Función para categorizar columnas
    if pd.api.types.is_numeric_dtype(column):
        return pd.Categorical(column)
    return column

# Funciones para traducción de datos

def traducir_texto(texto, destino='en'):     # Función para traducir texto
    translator = Translator()
    traduccion = translator.translate(texto, dest=destino)
    return traduccion.text

# Funciones para StreamLit

def reproducir(texto):  # función texto a audio
    global engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)
    engine.setProperty('volume', 0.9)
    engine.setProperty('voice', 'spanish')
    engine.say(texto)
    engine.runAndWait()


def silenciar():  # función para silenciar el audio
    global engine
    engine = pyttsx3.init()
    engine.setProperty('volume', 0.0)


def gpt3(usuario_input):  # función para llamar a la api de OpenAI usando GPT-3

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en videojuegos"},
            {"role": "user", "content": usuario_input}
        ],
        max_tokens=3000,
        temperature=0.5
    )
    respuesta = response['choices'][0]['message']['content']

    return respuesta


def reconocer_audio():   # función audio a texto 
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Escuchando...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            texto = recognizer.recognize_google(audio, language="es-ES")
            st.success(f"Texto reconocido: {texto}")
            return texto
        except sr.UnknownValueError:
            st.warning("No se pudo reconocer el audio.")
        except sr.RequestError as e:
            st.error(f"Error en la solicitud al servicio de reconocimiento de voz: {e}")




