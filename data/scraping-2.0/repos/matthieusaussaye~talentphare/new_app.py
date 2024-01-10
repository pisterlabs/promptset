#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:42:22 2023

@author: paulm
"""

import streamlit as st

import os
from st_custom_components import st_audiorec
import time

import wave
import io
import json
from typing import List
import requests as r
import base64
import mimetypes
import openai
openai.api_key = "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"
from audio_recorder_streamlit import audio_recorder


def bytes_to_wav(audio_bytes, output_filename, sample_width=2, frame_rate=44100, channels=2):
    with wave.open(output_filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(audio_bytes)
# App title
st.set_page_config(page_title="Job interview Chatbot")



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello, who are you and what job are you applying to ?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content":  "Hello, who are you and what job are you applying to ?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating response
def generate_response(prompt_input):
    string_dialogue = "You are a job interviewer. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. Ask question to 'User' related to its answer"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    #output = replicate.run(llm, 
                           #input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                           #       "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    response=openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                          messages=[dict_message])
    return response

audio_bytes = audio_recorder()

if audio_bytes is not None:
    
    bytes_to_wav(audio_bytes, 'output.wav')  # Replace audio_bytes with your audio data
    
    # The name of the .wav file
    filename = 'output.wav'
    
    # Open the .wav file
    wav_audio_data = open(filename, "rb")
   
    transcript = openai.Audio.transcribe("whisper-1", wav_audio_data)
    st.session_state.messages.append({"role": "user", "content": transcript["text"]})
    with st.chat_message("user"):
        st.write(transcript["text"])
if prompt := st.chat_input(True):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)