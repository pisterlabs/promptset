""" This file contains the code for the co-writer page.  This page allows
the user to chat with Luke Combs and receive guidance on their song writing.
This can be in the form of text or audio."""
import logging
from io import BytesIO
import wave
import base64
import os
import asyncio
import librosa
import numpy as np
import openai
import streamlit as st
from utils.model_utils import (
    get_inputs_from_llm, get_audio_sample
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "similar_clips" not in st.session_state:
    st.session_state.similar_clips = []
if "current_audio_clip" not in st.session_state:
    st.session_state.current_audio_clip = None
if "original_audio_clip" not in st.session_state:
    st.session_state.original_audio_clip = None
if "current_audio_string" not in st.session_state:
    st.session_state.current_audio_string = None
if "inputs" not in st.session_state:
    st.session_state.inputs = None

if not st.session_state.chat_history:
    st.success("**Welcome to co-writer!  There are two options for the chat:**\
                The first option is a standard text chat back and forth where\
                Luke can help you brainstorm ideas for your song.  The second\
                allows you to request that Luke help you compose an audio clip\
                based on your chat history and ideas generated.  You may switch\
                back and forth at any time to test them out!  Check out the sidebar\
                to make your selection.")
    st.markdown("---")
    with st.chat_message("assistant", avatar="🎸"):
        st.markdown("Hello, friend.  I'm excited to get our co-writing\
                    session started!  Why don't you tell me a little bit\
                    about the song you are working on?")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🎸"):
            st.markdown(message["content"])

# @TODO Create an "agent" LLM that can be used to pick up on the user's
# intent and then choose the appropriate model to respond with, i.e.
# music gen, encode / decode, or standard chat
response_type = st.sidebar.radio("Choose your response type",
                                ("Standard Chat", "Musical Chat"))
if st.session_state.current_audio_clip:
    st.write("Current audio clip: ")
    st.audio(np.array(st.session_state.current_audio_clip), sample_rate=32000)
    st.session_state.use_audio = st.sidebar.radio("Use current audio clip as input?", ("Yes", "No"))

def get_text_response(artist:str = "Dave"):
    """ Get a response from the artist in the form of text."""
    openai.api_key = os.getenv("OPENAI_KEY2")
    openai.organization = os.getenv("OPENAI_ORG2")
    if prompt := st.chat_input(f"Your message for {artist}:", key="chat_input1"):
        with st.spinner(f"{artist} is writing..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar = "🎸"):
                messages = [
                    {
                        "role": "system", "content": f"""You are Dave Matthews, the famous artist,
                        engaging in a co-writing session with a fellow musician the goal is to make
                        it as much like an actual co-writing session with Dave.  You should try to
                        take on his personality when answering, and do not break character.
                        Their latest question is {prompt} and your chat history
                        is {st.session_state.chat_history}. Continually gauge the tone of the question,
                        and if based on the chat history as well you think the user is asking for
                        a song lyric, feel free to respond with one.  The goal is to be interactive,
                        engaging, empathetic, and helpful.  Keep the conversation going until it
                        is clear the user is ready to end the chat.
                        """
                    },
                    {
                        "role": "user", "content": f"""Please answer my {prompt} about 
                        song writing."""
                    },
                ]
                message_placeholder = st.empty()
                full_response = ""
                # Set list of models to iterate through
                models = ["gpt-4-0613", "gpt-4", "ft:gpt-3.5-turbo-0613:david-thomas::7wEhz4EL"]
                for model in models:
                    try:
                        for response in openai.ChatCompletion.create(
                            model=model,
                            messages = messages,
                            max_tokens=300,
                            frequency_penalty=0.75,
                            presence_penalty=0.75,
                            temperature=1,
                            n=1,
                            stream=True
                        ):
                            full_response += response.choices[0].delta.get("content", "")
                            message_placeholder.markdown(full_response + "▌")
                            if response.choices[0].delta.get("stop"):
                                break
                        break
                    except TimeoutError as e:
                        logging.log(logging.ERROR, e)
                        continue
            message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

async def get_music_response():
    """ Get a response from the artist in the form of an audio clip. """
    # Audio File Upload
    uploaded_file = st.sidebar.file_uploader("Upload your audio file", type=["mp3", "wav"])
    if uploaded_file:
        st.session_state.original_audio_clip = uploaded_file.read()
        audio_data = uploaded_file.getvalue()
        audio, sr = librosa.load(BytesIO(audio_data), sr=32000)
        st.audio(st.session_state.original_audio_clip, format="audio/mp3", start_time=0)
        if prompt := st.chat_input("Your message for Dave:", key="chat_input_music"):
            with st.spinner("Dave is composing... This will take a minute"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar = "🎸"):
                    message_placeholder = st.empty()
                    full_response = ""
                    # @TODO Specially train a model to get the llm inputs
                    inputs = await get_inputs_from_llm()
                    st.session_state.inputs = inputs
                    #input_audio = await chunk_and_encode_encodec(audio)
                    #output = await get_audio_sample(inputs, input_audio)          
                    #st.markdown("**Current audio sample:**")
                    #message_placeholder.audio(np.array(output), sample_rate=32000)
                    #st.session_state.current_audio_clip = output
            
                    # Step 1: Convert NumPy array to byte stream
                    #byte_stream = BytesIO()

                    # Prepare wave file settings
                    #n_channels = 1
                    #sampwidth = 2  # Number of bytes
                    #framerate = 32000
                    #n_frames = len(output)

                    #with wave.open(byte_stream, 'wb') as wav_file:
                    #    wav_file.setnchannels(n_channels)
                    #    wav_file.setsampwidth(sampwidth)
                    #    wav_file.setframerate(framerate)
                    #    wav_file.writeframes(np.array(output).astype(np.int16).tobytes())

                    # Step 2: Base64 encode the byte stream
                    #byte_stream.seek(0)
                    #base64_audio = base64.b64encode(byte_stream.read()).decode('utf-8')
                    #st.session_state.current_audio_string = base64_audio

                st.session_state.chat_history.append({"role": "assistant",
                                                    "content": full_response})
        if st.session_state.inputs:
            st.write(st.session_state.inputs)       
    else:
        if prompt := st.chat_input("Your message for Dave:", key="chat_input_music"):
            with st.spinner("Dave is composing... This will take a minute"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar = "🎸"):
                    message_placeholder = st.empty()
                    full_response = ""
                    inputs = await get_inputs_from_llm()
                    if st.session_state.current_audio_string is not None and st.session_state.use_audio == "Yes":
                        audio = st.session_state.current_audio_string
                    else:
                        audio = None
                    output = await get_audio_sample(inputs, audio)
                    if output:
                        st.markdown("**Current audio sample:**")
                        message_placeholder.audio(np.array(output), sample_rate=32000)
                        st.session_state.current_audio_clip = output
                        # Step 1: Convert NumPy array to byte stream
                        byte_stream = BytesIO()

                        # Prepare wave file settings
                        n_channels = 1
                        sampwidth = 2  # Number of bytes
                        framerate = 32000

                        with wave.open(byte_stream, 'wb') as wav_file:
                            wav_file.setnchannels(n_channels)
                            wav_file.setsampwidth(sampwidth)
                            wav_file.setframerate(framerate)
                            wav_file.writeframes(np.array(output).astype(np.int16).tobytes())
                    
                        # Step 2: Base64 encode the byte stream
                        byte_stream.seek(0)
                        base64_audio = base64.b64encode(byte_stream.read()).decode('utf-8')
                        st.session_state.current_audio_string = base64_audio


                st.session_state.chat_history.append({"role": "assistant",
                                                    "content": full_response})

# Create a button to reset the chat history
reset_button = st.sidebar.button("Reset Chat History", type="primary", use_container_width=True)
if reset_button:
    st.session_state.chat_history = []
    st.experimental_rerun()

if response_type == "Standard Chat":
    get_text_response()
else:
    asyncio.run(get_music_response())
           