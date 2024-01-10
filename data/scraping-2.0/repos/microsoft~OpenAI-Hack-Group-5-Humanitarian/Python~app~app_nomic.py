
import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import streamlit as st

from st_custom_components import st_audiorec
#from audio_recorder_streamlit import audio_recorder
from scipy.io.wavfile import write
from footer import footer

import whisper
import warnings
warnings.filterwarnings('ignore')

import requests
import json
import pandas as pd
from langchain.llms import OpenAI
from gtts import gTTS

os.environ["OPENAI_API_KEY"] = "<your-api-key>"
model = whisper.load_model("base")
df = pd.read_csv('../mortality_data.csv')
# concatenate values from first 5 rows of column 'text'
train_data = df['TextValue'][0:5].str.cat(sep='. ')
llm = OpenAI(temperature=0)
language = 'en'


if __name__ == "__main__":

    st.title("AI-Powered QA Voice Assistant")

    st.markdown("I can answer questions about the WHO child mortality data.")
    st.markdown("Please ask your question in the text box below")
    
    question = st.text_input('Question', '')
    if question:
        prompt = f"""
Use only the dataset provided to answer the question. 

Dataset: {train_data} 

Question: {question}"""

        output = llm(prompt)
        st.markdown(f"The LLM response is: {output}")

        st.markdown("Converting response to speech...")
        myobj = gTTS(text=output, lang=language, slow=False)
        myobj.save("response.mp3")
        st.audio("response.mp3")
    
    footer()
