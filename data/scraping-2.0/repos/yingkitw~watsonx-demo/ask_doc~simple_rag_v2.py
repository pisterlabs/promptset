import sys
import logging
import os
import tempfile
import pathlib

import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

from genai.credentials import Credentials
from genai.schemas import GenerateParams
from genai.model import Model

from typing import Literal, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
# Most GENAI logs are at Debug level.
# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="技术支持助理",
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

st.header("技术支持助理 watsonx.ai 💬")
# chunk_size=1500
# chunk_overlap = 200

load_dotenv()

api_key = st.secrets["GENAI_KEY"]
api_endpoint = st.secrets["GENAI_API"]

api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

# handler = StdOutCallbackHandler()

creds = Credentials(api_key,api_endpoint)

params = GenerateParams(
    decoding_method="greedy",
    max_new_tokens=1000,
    min_new_tokens=1,
    # stream=True,
    top_k=50,
    top_p=1
)

# Sidebar contents
with st.sidebar:
    st.title("技术支持助理")
    uploaded_files = st.file_uploader("上传一个PDF文档", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files,chunk_size =250,chunk_overlap=20):
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
        # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name
            with st.spinner('Waiting for the file to upload'):
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

def querypdf(informations, question):
    # prompt = f"""
    # answer the question in 5 sentences base on the informations:
    # informations:
    # {informations}
    # question:
    # {question}
    # answer in point form:"""

    prompt = f"""[INST]作为一个工程师，请根据提供的白皮书回答。
    <<SYS>>
    白皮书:
    {informations}
    <<SYS>>
    问题:
    {question}
    [/INST]
    回答:"""

    prompts = [prompt]
    answer = ""
    for response in model.generate_async(prompts,ordered=True):
        answer += response.generated_text
    return answer

docs = read_pdf(uploaded_files)
if docs is not None:
    db = read_push_embeddings(docs)

model = Model(model="meta-llama/llama-2-70b-chat",credentials=creds, params=params)

with st.chat_message("system"):
    st.write("请输入你的查询")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("your query"):
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner(text="In progress...", cache=False):
        docs = db.similarity_search(query)
        answer = querypdf(docs, query)

    st.session_state.messages.append({"role": "agent", "content": answer}) 

    with st.chat_message("agent"):
        st.markdown(answer)