from pathlib import Path
from OpenAI_Training.config import get_api_key
from openai import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_api_key()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world! This is a streaming test.",
)

response.stream_to_file("output.mp3")
############
# Show to the screen
# App Framework
st.title('Boiler Plate:')
st.audio("speech.mp3")