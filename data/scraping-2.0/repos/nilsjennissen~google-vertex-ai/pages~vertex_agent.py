'''
This is the file for the google_vertex_agent
'''

#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import streamlit as st
from credentials import OPENAI_API_KEY, project_id

from main import rec_streamlit, speak_answer, get_transcript_whisper, get_transcript_google

import os
import google.generativeai as palm
import requests
import time
import urllib.parse
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import vertexai
from vertexai.preview.language_models import ChatModel


#%% -----------------------------------------  GOOGLE VERTEX AI ------------------------------------------------------#

# Initialise the vertexai environment
vertexai.init(project=project_id, location="us-central1")

# Initialise the chat model
model = ChatModel.from_pretrained("chat-bison@001")
chat = model.start_chat(examples=[])

#%% ---------------------------------------------  INTERFACE  --------------------------------------------------------#

# --------------------  SETTINGS  -------------------- #
st.set_page_config(page_title="Home", layout="wide")
st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""", unsafe_allow_html=True)

# --------------------- HOME PAGE -------------------- #
st.title("GOOGLE VERTEX AI")
st.write("""Chat with Google Vertex AI's PALM2 Bison Model""")
st.write("Let's start interacting with Vertex AI")


# ----------------- SIDE BAR SETTINGS ---------------- #
st.sidebar.subheader("Settings:")
tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)

# ------------------ FILE UPLOADER ------------------- #
st.sidebar.subheader("File Uploader:")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["csv", "html", "css", "py", "pdf", "ipynb"],
                                          accept_multiple_files=True)
st.sidebar.metric("Number of files uploaded", len(uploaded_files))
st.sidebar.color_picker("Pick a color for the answer space", "#C14531")

# Initialize docsearch as None
docsearch = None

# --------------------- USER INPUT --------------------- #
user_input = st.text_area("")
# If record button is pressed, rec_streamlit records and the output is saved
audio_bytes = rec_streamlit()

# ------------------- TRANSCRIPTION -------------------- #
if audio_bytes or user_input:

    if audio_bytes:
        try:
            with open("audio.wav", "wb") as file:
                file.write(audio_bytes)
        except Exception as e:
            st.write("Error recording audio:", e)
        transcript = get_transcript_google("audio.wav")
    else:
        transcript = user_input

    st.write("**Recognized:**")
    st.write(transcript)

    if any(word in transcript for word in ["abort recording"]):
        st.write("... Script stopped by user")
        exit()


    # ----------------------- ANSWER ----------------------- #
    with st.spinner("Fetching answer ..."):
        time.sleep(6)

    response = chat.send_message(transcript)
    answer = response.text
    st.write(answer)
    speak_answer(answer, tts_enabled)
    st.success("**Interaction finished**")


