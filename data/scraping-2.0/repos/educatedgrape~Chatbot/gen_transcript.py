from pytube import YouTube
import openai
import streamlit as st 

@st.cache_data
def generate_transcript(video_url):
    audio_file = YouTube(video_url).streams.filter(only_audio=True).first().download(filename="audio.mp4")
    file = open(audio_file, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file)

    return transcription