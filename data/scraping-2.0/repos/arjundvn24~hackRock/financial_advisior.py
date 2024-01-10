import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import webbrowser
# Set page config
from pvrecorder import PvRecorder
import subprocess
import sys
import whisper
from typing import List
from pathlib import Path
from collections import deque
import urllib.request
import time
import threading
import queue
import logging.handlers
import logging
import av
import numpy as np
import pydub
from twilio.rest import Client
import wave
import struct
import os
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import csv
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import Apifile
# # Set OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = Apifile.API
path = "audio_recording.mp3"
audio = []
# # Create Streamlit app
# st.title("Chatbot App")

# # Input field for user query
# query = st.text_input("Enter your query:")

# # Load the text document
# loader = TextLoader('stocks.csv')

# # Create an instance of ChatOpenAI
# chat_openai_instance = ChatOpenAI()

# # Create an index from the loader
# index = VectorstoreIndexCreator().from_loaders([loader])

# # Check if query is not empty
# if query:
#     # Perform query and get the response
#     response = index.query(query, chat_openai_instance)

#     # Display the response
#     st.write("Chatbot Response:")
#     st.write(response)


audio_filename = "audio_recording.mp3"

model = whisper.load_model("tiny")


HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def translate(audio):
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model.transcribe(audio_filename, **translate_options)
    return result


def make_audio(recorder):
    status_indicator = st.empty()
    # stop_button = st.button("Stop Recording")
    # status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    from IPython.display import Audio

    audio_filename = "audio_recording.mp3"
    Audio(audio_filename)
    try:
        recorder.start()

        start_time = time.time()

        while True:
            frame = recorder.read()
            audio.extend(frame)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 10:
                text_output.text("stopped")  # Stop recording after 5 seconds
                break
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        recorder.delete()

        with wave.open(path, "w") as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
        print(f"Recording saved as {path}")
    result = translate(audio_filename)

    text_output.text(result["text"])


def main():
    devices = PvRecorder.get_available_devices()
    recorder = PvRecorder(device_index=1, frame_length=512)
    audio = []

    st.header("Real Time Speech-to-Text")
    st.markdown(
        """
This demo app is using [Willow Model],
an open speech-to-text engine.
"""
    )

    sound_only_page = "Sound only (sendonly)"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page])

    if app_mode == sound_only_page:
        if st.button("Start Recording"):
            make_audio(recorder)


if __name__ == '__main__':
    st.set_page_config(
        page_title="What's TRUE: Financial Advisor",
        page_icon="💼",
        layout="wide",
    )

# Custom CSS to style the app
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://example.com/finance_background_image.jpg');  /* Replace with your image URL */
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .title-text {
            font-size: 36px;
            color: #ffffff;  /* White text color */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌟 What's TRUE: Your Financial Advisor")
    st.markdown("Aladin + Us is the best solution you've got! 💰",
                unsafe_allow_html=True)

    API = "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"
    user_question = st.text_input(
        "Please enter your age, financial goals, short term and long term goal", value="")

    llm = OpenAI(temperature=0.7, openai_api_key=API)

    if st.button("Tell me about it", key="tell_button"):
        template = "{question}\n\n"
        prompt_template = PromptTemplate(
            input_variables=["question"], template=template)
        question_chain = LLMChain(llm=llm, prompt=prompt_template)
        st.subheader("Result 1")
        st.info(question_chain.run(user_question))

        template = "Here is a statement:\n{statement}\nMake a bullet point list of the assumptions you made when producing the above statement.\n\n"
        prompt_template = PromptTemplate(
            input_variables=["statement"], template=template)
        assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)
        assumptions_chain_seq = SimpleSequentialChain(
            chains=[question_chain, assumptions_chain], verbose=True
        )
        st.subheader("Result 2")
        st.markdown(assumptions_chain_seq.run(user_question))

        # Chain 3
        template = "Here is a bullet point list of assertions:\n{assertions}\nFor each assertion, determine whether it is true or false. If it is false, explain why.\n\n"
        prompt_template = PromptTemplate(
            input_variables=["assertions"], template=template)
        fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)
        fact_checker_chain_seq = SimpleSequentialChain(
            chains=[question_chain, assumptions_chain,
                    fact_checker_chain], verbose=True
        )
        st.subheader("Result 3")
        st.markdown(fact_checker_chain_seq.run(user_question))

        # Final Chain
        template = "In light of the above facts, how would you answer the question '{}'\n".format(
            user_question)
        template = "{facts}\n" + template
        prompt_template = PromptTemplate(
            input_variables=["facts"], template=template)
        answer_chain = LLMChain(llm=llm, prompt=prompt_template)
        st.subheader("Final Result")
        overall_chain = SimpleSequentialChain(
            chains=[question_chain, assumptions_chain,
                    fact_checker_chain, answer_chain],
            verbose=True,
        )

        st.success(overall_chain.run(user_question))
    main()
