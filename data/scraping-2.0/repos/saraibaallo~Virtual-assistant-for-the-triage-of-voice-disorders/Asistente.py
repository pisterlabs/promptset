import streamlit as st
import openai
from openai.error import APIError, InvalidRequestError, OpenAIError, APIConnectionError
import base64
import pygame
import Mensajes
from gtts import gTTS
from gtts.tts import gTTSError

def autoplay_audio(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}"></audio>'
    st.markdown(audio_tag, unsafe_allow_html=True)

# Conversión de texto a audio y reproducción del mismo (el programa no continuará hasta que se haya terminado de reproducir)
def txt_audio(texto,gpt):
    pygame.init()   #iniciamos el reproductor de audio
    try: 
        tts = gTTS(text=texto, lang="es", slow=False)
        tts.save(f'sound_gtts.wav')
        sonido = pygame.mixer.Sound(f'sound_gtts.wav')
        sonido.play()
    except gTTSError:
        msj = "No es posible generar el audio del mensaje, por ahora continuaremos la consulta con el texto."
        Mensajes.mensaje_doctor(msj)

    # Esperar a que termine la reproducción
    while pygame.mixer.get_busy():
        pass
    
# Comunicación con ChatGPT
def ask_chatgpt(prompt):
    with st.spinner('Espere un momento...'):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "set language es"},
                {"role": "system", "content": "Eres un amable y útil doctor en español."},
                {"role": "user", "content": prompt}],
                max_tokens = 500
            )
            answer = response['choices'][0]['message']['content']

        except (APIError, InvalidRequestError, OpenAIError, APIConnectionError, Exception):
             answer = "Lo siento, ahora mismo no es posible responder."
    return answer