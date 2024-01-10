import streamlit as st
import openai
import whisper
import os

# Configura aquí tu clave API de OpenAI
openai.api_key = 'my-key'
model = whisper.load_model("base")

def transcribe_audio(model, audio_file):
    # Guardar el archivo de audio temporalmente
    temp_file_path = "temp_audio." + audio_file.name.split('.')[-1]
    with open(temp_file_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Transcribir el audio
    transcript = model.transcribe(temp_file_path)

    # Eliminar el archivo temporal
    os.remove(temp_file_path)

    return transcript['text']

def CustomChatGPT(user_input):
    messages = [{"role": "system", "content": "You are an office administrator, summarize the text in key points"}]
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    return ChatGPT_reply

# Interfaz de usuario de Streamlit
st.title('Evidencia de Competencias NLP: Resumen de Audio - Alejandro Murcia')

uploaded_file = st.file_uploader("Carga un archivo de audio", type=['mp3', 'wav', 'm4a'])

if uploaded_file is not None:
    if st.button('Transcribir Audio'):
        st.text("Transcribiendo...")
        transcription = transcribe_audio(model, uploaded_file)
        st.text_area("Transcripción:", transcription, height=150)
        st.session_state['transcription'] = transcription

    if st.button('Resumir Texto'):
        if 'transcription' in st.session_state and st.session_state['transcription']:
            st.text("Resumiendo...")
            summary = CustomChatGPT(st.session_state['transcription'])
            st.text_area("Resumen:", summary, height=150)
        else:
            st.warning("Primero transcribe un audio para resumir.")
