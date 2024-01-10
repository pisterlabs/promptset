import os
import tempfile
from dotenv import load_dotenv

import openai
from elevenlabs import set_api_key

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import streamlit as st
from audiorecorder import audiorecorder

load_dotenv()

set_api_key(os.environ['ELEVEN_LABS_API_KEY'])
openai.api_key = os.environ['OPENAI_API_KEY']

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(temp_audio.name)
    
    return transcript["text"].strip()


meal_template = PromptTemplate(
    input_variables=["ingredients"],
    template="""Give me an example of 2 meals that could 
                be made using only the following ingredients: {ingredients}
                """,
)

llm = OpenAI(temperature=0.9)

meal_chain = LLMChain(
    llm=llm,
    prompt=meal_template,
    verbose=True
)

st.title("Meal planner")

user_prompt = st.text_input("A comma-separated list of ingredients")
audio = audiorecorder("Click to record", "Click to stop recording")

if audio:
    transcript = transcribe_audio(audio.export().read())
    st.write(transcript)
    with st.spinner("Generating..."):
        output = meal_chain.run(user_prompt)
        st.write(output)        

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = meal_chain.run(user_prompt)
        st.write(output)        