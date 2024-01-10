
import os
import numpy as np
import streamlit as st
from io import BytesIO
import streamlit.components.v1 as components
import streamlit as st

from st_custom_components import st_audiorec
from scipy.io.wavfile import write
from footer import footer

#import whisper
import warnings
warnings.filterwarnings('ignore')

import requests
import json
import pandas as pd
import faiss
import pickle
from langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from gtts import gTTS

os.environ["OPENAI_API_KEY"] = "<your-api-key>"

index = faiss.read_index("docs.index")
with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
language = 'en'


if __name__ == "__main__":

    st.title("AI-Powered QA Voice Assistant")

    st.markdown("I can answer questions about the WHO child mortality data.")
    st.markdown("Please ask your question in the text box below")
   
    question = st.text_input('Question', '')
    if question:
        
        output = chain(question)
        st.markdown(f"The LLM response is: {output}")

        st.markdown("Converting response to speech...")
        myobj = gTTS(text=output['answer'], lang=language, slow=False)
        myobj.save("response.mp3")
        st.audio("response.mp3")
    
    footer()
