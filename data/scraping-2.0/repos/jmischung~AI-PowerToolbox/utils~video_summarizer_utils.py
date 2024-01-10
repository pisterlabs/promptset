import os
from pathlib import Path
import openai
import streamlit as st
from pytube import YouTube

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set paths
PROJECT_ROOT = Path(__file__).parent.parent
TEMP_DIR = PROJECT_ROOT / 'temp'


def get_audio_youtube(url):
    # Fetch YouTube video
    yt = YouTube(url)

    # Get medium quality audio stream (assuming a middle bitrate represents "medium" quality)
    audio_streams = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr')
    medium_quality_stream = audio_streams[len(audio_streams) // 2]

    # Download the audio stream
    medium_quality_stream.download(filename=TEMP_DIR / 'audio.mp4')


def transcribe_audio(audio_file_path):
    # Transcribe audio
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", file=audio_file)

    # Delete audio file
    os.remove(audio_file_path)

    return transcription['text']


def key_points_extraction(transcription):
    # Extract key points from transcription
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": """
                    You are an expert summarizer with a specialty in distilling information into a
                    concise list of key points. Based on the following text, identify and list only
                    the key points that were discussed or brought up. These should be only the most
                    important ideas, findings, or topics that are crucial to the essence of the
                    discussion. Your goal is to provide a list of a maximum of seven key points
                    that someone could read to quickly understand what the discussion was about.
                    The fewer key points you provide, the better. If you are unsure about a point,
                    do not include it. If you are unsure about whether a point is a key point,
                    do not include it. Return your list in markdown format with each key point on a
                    new line as a bullet point and the H3 header "Key Points" at the top.
                """
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )

    # Return key points
    return response['choices'][0]['message']['content']
