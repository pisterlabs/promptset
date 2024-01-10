import sys
import logging
import os
import tempfile
import pathlib

import streamlit as st
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# from genai.credentials import Credentials
# from genai.schemas import GenerateParams
# from genai.model import Model

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
    page_title="技术支持",
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


st.header("技术支持")
# chunk_size=1500
# chunk_overlap = 200

load_dotenv()

api_key = os.getenv("API_KEY", None)
project_id = os.getenv("PROJECT_ID", None)

# handler = StdOutCallbackHandler()

creds = {
    "url"    : "https://us-south.ml.cloud.ibm.com",
    "apikey" : api_key
}

params = {
    GenParams.DECODING_METHOD:"greedy",
    GenParams.MAX_NEW_TOKENS:1000,
    GenParams.MIN_NEW_TOKENS:1,
    GenParams.TEMPERATURE:0.5,
    GenParams.TOP_K:50,
    GenParams.TOP_P:1
}

# Sidebar contents
with st.sidebar:
    st.title("技术支持")
    uploaded_files = st.file_uploader("上传一个PDF文档", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files,chunk_size =250,chunk_overlap=20):
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
        # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name
            with st.spinner('请上传PDF文档'):
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

def querypdf(informations, history, question):
    # prompt = f"""
    # answer the question in 5 sentences base on the informations:
    # informations:
    # {informations}
    # question:
    # {question}
    # answer in point form:"""

    prompt = f"""[INST]作为一个技术支持工程师，请根据提供的白皮书用中文回答。
    -只根据白皮书内容回答，说明信息来源。
    -如果不知道，不要猜，就说不知道，并请客户查看官网信息。
    <<SYS>>
    白皮书:
    {informations}
    前面提问:
    {history}
    <<SYS>>
    问题:
    {question}
    [/INST]
    回答:"""

    prompts = [prompt]
    answer = ""
    for response in model.generate_text(prompts):
        answer += response.replace("\\n\\n","\n")
    return answer

docs = read_pdf(uploaded_files)
if docs is not None:
    db = read_push_embeddings(docs)

model = Model("meta-llama/llama-2-70b-chat",creds, params, project_id)
# model = Model(model="meta-llama/llama-2-70b-chat",credentials=creds, params=params)

history = []

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

    history += [query]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner(text="正在查询...", cache=False):
        docs = db.similarity_search(query)
        answer = querypdf(docs, history, query)

    st.session_state.messages.append({"role": "agent", "content": answer}) 

    with st.chat_message("agent"):
        st.markdown(answer)