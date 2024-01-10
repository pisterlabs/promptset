import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceTextGenInference

from langchain.llms import VertexAI

from elasticsearch import Elasticsearch, helpers

@st.cache_resource
def get_es():
    es_server = st.secrets["GOVES_SERVER"]
    es_username = st.secrets["GOVES_USERNAME"]
    es_password = st.secrets["GOVES_PASSWORD"]
    url = f"https://{es_username}:{es_password}@{es_server}:443"
    return Elasticsearch([url], verify_certs=True) 


@st.cache_resource
def load_vertexai():
    return VertexAI()


@st.cache_resource
def load_llama2_llm():
    HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    LLAMA2_HF_URL = st.secrets["LLAMA2_HF_URL"]

    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    server_kwargs = {"headers": headers}
    llm = HuggingFaceTextGenInference(
        inference_server_url = LLAMA2_HF_URL,
        temperature=0.1,
        top_k=30,
        # do_sample=True,
        max_new_tokens=512,
        timeout=120,
        server_kwargs = server_kwargs
    )
    return llm


# @st.cache_resource
# def getLlama2Client():
#     HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
#     LLAMA2_HF_URL = st.secrets["LLAMA2_HF_URL"]
#     client = Client(
#         LLAMA2_HF_URL, 
#         headers={
#             "Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"
#             },
#         timeout=120,
#     )
#     return client


@st.cache_resource
def load_openai_llm():
    openai_type = st.secrets["OPENAI_TYPE"]
    print(f"-- Initializing connection to OpenAI with {openai_type}")
    # Set OpenAI API key from Streamlit secrets
    
    
    if openai_type == "proxy":
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        openai.api_base = st.secrets["OPENAI_API_BASE"]
        default_model = "gpt-3.5-turbo"
        openai.default_model = default_model
        openai.default_engine = None
        return ChatOpenAI(
            temperature=0.3,
            model=default_model)
    if openai_type == "openai":
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        default_model = "gpt-3.5-turbo"
        openai.default_model = default_model
        openai.default_engine = None
        return ChatOpenAI(
            temperature=0.3,
            model=default_model)
    if openai_type == "azure":
        openai.api_type = "azure"
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        openai.api_base = st.secrets["OPENAI_API_BASE"]
        default_model = st.secrets["OPENAI_DEPLOYMENT_NAME"]
        openai.default_model = default_model
        openai.default_engine = default_model
        openai.api_version = st.secrets["OPENAI_API_VERSION"]
        # return ChatOpenAI(
        #     engine = openai.default_engine,
        #     temperature=0.3,
        #     model=default_model)
        return ChatOpenAI(
            model_name=default_model,
            model_kwargs={"deployment_id":default_model}
        )
    else:
        raise Exception(f"Error configuring llm, type from config: {openai_type}")
