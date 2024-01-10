import openai
import streamlit as st
import dart_fss as dart
import os
import vertexai

from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool


# Open DART API KEY 설정
api_key=os.getenv('DART_API_KEY')
dart.set_api_key(api_key=api_key)

# Set up the Vertex AI client
PROJECT_ID = os.getenv("PROJECT_ID")
vertexai.init(project=PROJECT_ID, location="us-central1")

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=2048,
    temperature=0,
    top_p=0.8,
    top_k=40,
)

# Load the company list
dart.get_corp_list()



with st.sidebar:
    st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if "chatbot_api_key" not in st.session_state:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = st.session_state["chatbot_api_key"]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)