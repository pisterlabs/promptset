import sys
import logging
import os
import tempfile
import pathlib
import json

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
    page_title="quiz builder",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
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


st.header("quiz builder")
# chunk_size=1500
# chunk_overlap = 200

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
    GenParams.DECODING_METHOD:"sample",
    GenParams.MAX_NEW_TOKENS:1000,
    GenParams.MIN_NEW_TOKENS:1,
    GenParams.TEMPERATURE:0.7,
    GenParams.TOP_K:50,
    GenParams.TOP_P:1,
}

# Sidebar contents
with st.sidebar:
    st.title("quiz builder")
    uploaded_files = st.file_uploader("upload PDF documents", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files,chunk_size =250,chunk_overlap=20):
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
        # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name
            with st.spinner('uploading PDF documents'):
                loader = PyPDFLoader(filepath)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
                docs = text_splitter.split_documents(data)
                return docs

def read_push_embeddings(docs):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    # embeddings = HuggingFaceEmbeddings()
    temp_dir = tempfile.TemporaryDirectory()
    db = Chroma.from_documents(docs, embeddings)
    return db

def buildquiz(informations, topic):
    prompt = f"""[INST]base on the topic and informations provided, 
    generate a scenario base question along with multiple choice quiz
    assume a role in the question, and assign a specific task to the role, and ask which answer best address the task.
    the answer should describe a operation very detail.
    the answer options should be specific, descriptive, and with more detail about the technique.
    mark the answer and provide explaination.
    <<SYS>>
    notes:
    - one paragraph per one answer option.
    - please follow the layout provided in backquoted.
    - ensure the answer options be different, but similar enough that the user hard to determine.
    - ensure only one answer option be correct.
    - explain the correct answer as well as the incorrect answer options.
    - output in markdown.
    topic:{topic}
    informations:
    {informations}
    layout: `question?

    a) answer option.\n
    b) answer option.\n
    c) answer option.\n
    d) answer option.\n

    correct answer (option), explaination.`
    <</SYS>>
    [/INST]
    markdown quiz:"""

    prompts = [prompt]
    answer = ""
    for response in model.generate_text(prompts):
        answer += response
    return answer

docs = read_pdf(uploaded_files)
if docs is not None:
    db = read_push_embeddings(docs)

model = Model("meta-llama/llama-2-70b-chat",creds, params, project_id)

history = []

with st.chat_message("system"):
    st.write("input your question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if topic := st.chat_input("your topic"):
    with st.chat_message("user"):
        st.markdown(topic)

    history += [topic]

    st.session_state.messages.append({"role": "user", "content": topic})
    with st.spinner(text="building...", cache=False):
        docs = db.similarity_search(topic)
        answer = buildquiz(docs,topic)
        # print(answer)
        st.session_state.messages.append({"role": "agent", "content": answer}) 

        with st.chat_message("agent"):
            st.markdown(answer)