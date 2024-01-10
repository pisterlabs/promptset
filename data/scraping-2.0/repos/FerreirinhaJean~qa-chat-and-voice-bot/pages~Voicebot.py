import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("API_KEY_OPENAI")

st.title("🎙️ 💬 QA With Voicebot")

"""
Este Voicebot tem como objetivo responder perguntas sobre os documentos vetorizados!
"""

audio_bytes = audio_recorder(text="", neutral_color="#FFF", pause_threshold=1)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Como posso ajudá-lo?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if audio_bytes:
    with open("audios/temp.wav", "wb") as temp_file:
        temp_file.write(audio_bytes)

    audio_file = open("audios/temp.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    st.session_state.messages.append({"role": "user", "content": transcript.text})
    st.chat_message("user").write(transcript.text)
