import streamlit as st
from audio_recorder_streamlit import audio_recorder
import openai
import numpy as np
from scipy.io.wavfile import write

openai.api_key = st.secrets["OPENAI_API_KEY"]
whisper_model_id = "text-davinci-002"

# Transcribe audio function
def transcribe_audio():
    audio_file = open("recorded_audio.wav", "rb")
    transcription = openai.Audio.transcribe(
        "whisper-1", audio_file
    )
    return transcription

# Analyze transcription function
def analyze_transcription(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant to retrieve important information from a given message. Each variable should have its own line."},
            {"role": "user", "content": "Retrieve name, issue and phone number from the following text: {}".format(transcription.text)}
            ],
        temperature=0.2
    )
    return response

# Front End
st.title('Speech to Text')

st.subheader("Please describe you issue and provide your name, and phone number.")

# Record the audio
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_bytes)

# Analyze audio
if audio_bytes is not None:
    transcription = transcribe_audio()
    transcription_text = transcription.text
    st.subheader(transcription_text)
    response = analyze_transcription(transcription)
    response_text = response.choices[0]["message"].content
    st.write(response_text)
