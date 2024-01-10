from __future__ import unicode_literals
import streamlit as st 
import json 
import os 
import time
import sys
from dotenv import load_dotenv
import requests
from pytube import YouTube
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
# from langchain.vectorstore import VectorStoreIndexCreator

from htmlTemplates import css, bot_template, user_template
import youtube_dl
import pafy
import langchain

st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")

st.markdown(""" <style>

MainMenu {visibility: hidden;}
header {visibility:hidden;}
footer {visibility: hidden;}
[data-testid="stAppViewContainer"]{
# background-image: url("https://tse4.mm.bing.net/th?id=OIP.j152CjiQZAA7hdfvkhqZBQHaE6&pid=Api&P=0");
background-color:cover;
}
</style> """, unsafe_allow_html=True)

langchain.verbose = False

load_dotenv()
api_token = 'API-Key'
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

base_url = "base_url"

headers = {
    "authorization": api_token,
    "content-type": "application/json"
}


import yt_dlp as youtube_dl
import os
import re

def clean_filename(filename):
    # Remove invalid characters from the filename
    cleaned_filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    return cleaned_filename
output_path='C:/Users/vijja/Desktop/Main Project'
def save_audio(url):
    try:
        options = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join('C:/Users/vijja/Desktop/Main Project', '%(main)s.%(ext)s'),
        }

        with youtube_dl.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            # video_title = clean_filename(info['title'])
            audio_path = os.path.join(output_path, "main.mp3")
            sample=os.rename("main.mp3", audio_path)
            print("Audio download complete!")
    except Exception as e:
        print("An error occurred while downloading the audio:", str(e))
    return Path('NA.mp3')


# Assembly AI speech to text
def assemblyai_stt(audio_filename):
    with open(audio_filename , "rb") as f:
        response = requests.post(base_url + "/upload",
                                headers=headers,
                                data=f)

    upload_url = response.json()["upload_url"]
    data = {
        "audio_url": upload_url
    }
    url = base_url + "/transcript"
    response = requests.post(url, json=data, headers=headers)
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
            break

        elif transcription_result['status'] == 'error':
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

        else:
            print("Processing...")
            time.sleep(3)
    print(transcription_result['text'])
    file = open('transcription.txt', 'w')
    file.write(transcription_result['text'])
    file.close()
    return transcription_result['text']

# Open AI code
def langchain_qa(query):
    loader = TextLoader('transcription.txt')
    index = VectorstoreIndexCreator().from_loaders([loader])
    query = query
    result = index.query(query)
    return result


#Streamlit Code
# st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")

# st.markdown(""" <style>

# MainMenu {visibility: hidden;}
# header {visibility:hidden;}
# footer {visibility: hidden;}
# [data-testid="stAppViewContainer"]{
# # background-image: url("https://tse4.mm.bing.net/th?id=OIP.j152CjiQZAA7hdfvkhqZBQHaE6&pid=Api&P=0");
# background-color:cover;
# }
# </style> """, unsafe_allow_html=True)


st.title("Chat with Your Audio using LLM")
st.write(css, unsafe_allow_html=True)
if "conversation" not in st.session_state:
        st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

    # st.header("Chat with multiple PDFs :books:")
user_question = st.text_input("Ask a query from your documents:")
if user_question:
    st.info("Your Query is: " + query)
    result = langchain_qa(query)
    st.success(result)

    
with st.sidebar:
    input_source = st.text_input("Enter the YouTube video URL")

    if input_source != None:
        col1, col2 = st.columns(2)

        with col1:
            st.info("Your uploaded video")
            if st.button('Get Video'):
                st.video(input_source)
            audio_filename = save_audio(input_source)
            transription = assemblyai_stt(audio_filename)
            st.info(transription)
        
        # with col2:
        #     st.info("Chat Below")
        #     query = st.text_area("Your Query....")
        #     if query is not None:
        #         if st.button("Ask"):
                    
                    
                    
               

