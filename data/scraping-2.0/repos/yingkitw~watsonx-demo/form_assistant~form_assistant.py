import sys
import logging
import os
import tempfile
import pathlib

import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from typing import Literal, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
# Most GENAI logs are at Debug level.
# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="form assistant",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 500px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.header("form assistant with watsonx.ai 💬")

load_dotenv()

api_key = st.secrets["API_KEY"]
project_id = st.secrets["PROJECT_ID"]

api_key = os.getenv("API_KEY", None)
project_id = os.getenv("PROJECT_ID", None)

creds = {
    "url"    : "https://us-south.ml.cloud.ibm.com",
    "apikey" : api_key
}

params = {
    GenParams.DECODING_METHOD:"greedy",
    GenParams.MAX_NEW_TOKENS:1000,
    GenParams.MIN_NEW_TOKENS:1,
    GenParams.REPETITION_PENALTY:1.1,
    GenParams.TOP_K:50,
    GenParams.TOP_P:1,
    GenParams.STOP_SEQUENCES:['end_of_form','<>','<EOS>'],
}

def buildjson(requirement):
    prompt = f"""[INST]
    <<SYS>>
    build a json structure for customer to input data for following requirement.
    - flatten the json structure.
    - show only name, value pair.
    - all field be empty value.
    - dont show notes.
    <</SYS>>
    requirements: {requirement}
    [/INST]flatten json:"""
    output = ""
    for response in model13b.generate_text([prompt]):
        output = response
    return output

def buildform(requirement, jsonform):
    prompt = f"""[INST]
    <<SYS>>
    build a html form that for customer to input value for following json base on the requirement.
    - for option, add a empty item if no value.
    - dont show JSON.
    - dont show note.
    - end with <EOS>
    <</SYS>>
    requirement: {requirement}
    json: `{jsonform}`
    [/INST]html form:"""
    output = ""
    for response in model13b.generate_text([prompt]):
        output = response
    return output.replace("<end_of_form>","")

def buildquestions(answerjson, requirement, lastmessage):
    prompt = f"""[INST]
    <<SYS>>you are customer service agent, generate message to guiding the user to fill the form with following steps:
    1. understand the answer and requriement provided in backquoted.
    2. check the answer json, gather a list the answer fields with empty value.
    3. if no field shortlisted in step 2, then just say thank you and skip step 4.
    4. for all field shortlisted in step 2, generate a concise question to request the answer.
    5. generate message with style similar to the last message provided in backquoted.
    note: 
    - empty value means empty string or zero or false or none.
    - dont repeat.
    - dont show note.
    - dont show (empty)
    - dont show json.
    <</SYS>>
    requirements: `{requirement}`
    answer: {answerjson}
    last message: `{lastmessage}`
    [/INST]response:"""

    # print("prompt:"+prompt)

    output = ""
    for response in model70b.generate_text([prompt]):
        output = response
    # print(output)
    return output.replace("<EOS>","")

def buildanswer(answer, existinganswer, jsonform):
    prompt = f"""[INST]
    <<SYS>>
    extract the answer in json from the answer to response to the json form.
    notes:
    - merge the answer with existing answer.
    - if not sure, dont guess, leave it empty.
    - show the merged answer only.
    - end with <EOS>
    <</SYS>>
    answers: {answer}
    existing answers: {existinganswer}
    json form: {jsonform}
    [/INST]merged answer:"""
    output = ""
    for response in model13b.generate_text([prompt]):
        output = response
    return output.replace("<EOS>","")

def fillform(answer, form):
    prompt = f"""[INST]
    <<SYS>>
    fill the html form with the answer value in json.
    - for option, add a empty item if no value. select empty item if the field has empty value.
    - dont show json.
    - dont show note.
    - end with <EOS>
    <</SYS>>
    answer: `{answer}`
    html form: {form}
    [/INST]html form with answer:"""

    print(prompt)

    output = ""
    for response in model70b.generate_text([prompt]):
        output = response
    return output.replace("<EOS>","")

# model = Model("mistralai/mistral-7b-instruct-v0-2",creds, params, project_id)
model70b = Model("meta-llama/llama-2-70b-chat",creds, params, project_id)
model13b = Model("meta-llama/llama-2-13b-chat",creds, params, project_id)


if "requirement" not in st.session_state:
    st.session_state.requirement = ""

if "jsonform" not in st.session_state:
    st.session_state.jsonform = ""

if "form" not in st.session_state:
    st.session_state.form = ""

if "filledform" not in st.session_state:
    st.session_state.filledform = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "lastmessage" not in st.session_state:
    st.session_state.lastmessage = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar contents
with st.sidebar:
    st.title("form assistant")

    btBuildForm = st.button("build form and guide me to fill")
    # btBuildQuestions = st.button("guide form filling with questions")
    # btFillForm = st.button("fill form")

st.session_state.requirement = st.text_area("requirement",height=10)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if btBuildForm:
    with st.spinner(text="building the form...", cache=False):
        jsonform = buildjson(st.session_state.requirement)
        form = buildform(st.session_state.requirement, st.session_state.jsonform)
        st.session_state.jsonform = jsonform
        st.session_state.form = form
        st.session_state.filledform = form
        st.session_state.answer = jsonform

    with st.chat_message("system"):
        with st.spinner(text="building the questions...", cache=False):
            questions = buildquestions("{}",st.session_state.requirement,"").replace('\\n', '\n').replace('\\t', '\t')
            st.session_state.lastmessage = questions
            st.markdown(questions)
            st.session_state.messages.append({"role": "agent", "content": questions})

if answer := st.chat_input("your answer"):
    with st.chat_message("user"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "user", "content": answer})
    with st.spinner(text="In progress...", cache=False):
        answerjson = buildanswer(answer, st.session_state.answer, st.session_state.jsonform)
        st.session_state.answer = answerjson

    with st.chat_message("system"):
        with st.spinner(text="building the questions...", cache=False):
            questions = buildquestions(st.session_state.answer,st.session_state.requirement,st.session_state.lastmessage).replace('\\n', '\n').replace('\\t', '\t')
            st.markdown(questions)
            st.session_state.messages.append({"role": "agent", "content": questions})

    filledform = fillform(st.session_state.answer, st.session_state.form)
    st.session_state.filledform = filledform

with st.sidebar:
    with st.container(border=True):
        st.components.v1.html(st.session_state.filledform,height=300)
    st.code(st.session_state.answer,language="json")