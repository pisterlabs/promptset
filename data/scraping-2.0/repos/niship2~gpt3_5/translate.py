import openai
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI
import streamlit as st
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

DEPLOYMENT_NAME = st.secrets["OPENAI_API_MODEL_NAME"]
BASE_URL = st.secrets["OPENAI_API_BASE"]
API_KEY = st.secrets["API_KEY"]


llm2 = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0.0,
    # streaming=True,
)


def translate_text(text: str) -> str:
    msg = "この文章を日本語に翻訳してください。" + text
    message = [HumanMessage(content=msg)]
    return llm2(message).content
