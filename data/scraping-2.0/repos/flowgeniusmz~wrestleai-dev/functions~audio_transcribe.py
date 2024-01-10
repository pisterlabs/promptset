import streamlit as st
from functions import audio_split
from openai import OpenAI

client = OpenAI(api_key=st.secrets.openai.api_key)
model = st.secrets.openai.model_whisper
responseformat = "text"

def TranscribeAudio(varAudioFilePath):
    transcript = ""
    audiochunks = audio_split.SplitAudio(varAudioFilePath=varAudioFilePath)
    for chunk in audiochunks:
        try:
            with open(chunk, "rb") as audiofile:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audiofile, 
                    response_format=responseformat
                )
                transcript += response + " "
        except Exception as e:
            print(f"An error occurred with chunk {chunk}: {e}")
    return transcript