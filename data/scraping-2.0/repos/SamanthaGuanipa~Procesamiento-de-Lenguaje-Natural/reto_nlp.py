import os
import certifi
import ssl
import streamlit as st
import json

# Set the REQUESTS_CA_BUNDLE environment variable
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import ssl

print(ssl.get_default_verify_paths())

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import openai

import whisper

 

# Replace 'YOUR_API_KEY' with your actual OpenAI API key

openai.api_key = "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"

st.title("Meeting Summary Generator")
st.subheader("This is summarized by Chat GPT")
st.write('\n\n')

file_path = st.text_input("File name","MA1.m4a")
 

def load_whisper_model():

    try:

        model = whisper.load_model("base")

        return model

    except Exception as e:

        print(f"Error loading Whisper model: {e}")

        return None

 

def transcribe_audio(model, file_path):

    try:

        transcript = model.transcribe(file_path)

        return transcript['text']

    except Exception as e:

        print(f"Error during transcription: {e}")

        return ""
def custom_chatgpt(user_input):
    messages = [{"role": "system", "content": "You are an office administrator, summarize the text in key points"}]

    messages.append({"role": "user", "content": user_input})

    try:

        response = openai.ChatCompletion.create(

            model="gpt-3.5-turbo",

            messages=messages

        )

        chatgpt_reply = response["choices"][0]["message"]["content"]

        return chatgpt_reply

    except Exception as e:

        print(f"Error in ChatGPT response: {e}")

        return ""
# Main Execution 
model = load_whisper_model()

if model:

    transcription = transcribe_audio(model, file_path)
    summary = custom_chatgpt(transcription)

if st.button('Generate Summary'):
    transcp_final = transcription
    response_final = summary
    st.write("The generated transcription is: " + transcp_final)
    st.write("The generated summary is: " + response_final)
else:
	st.write("Press the above button..")

