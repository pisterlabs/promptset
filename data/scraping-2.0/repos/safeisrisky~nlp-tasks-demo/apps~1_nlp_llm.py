import streamlit as st
import pandas as pd
import openai
import json
import utils
from utils import get_message
from utils import get_system_prompt, get_companies_prompt, get_non_companies_prompt

if not utils.check_password():
    st.stop()


st.title("NLP Tasks using LLM")
if "jsonresp" not in st.session_state:
    st.session_state.jsonresp = None
openai.api_key = st.secrets["OPENAI_API_KEY"]
app_password = st.secrets["APP_PASSWORD"]
client = openai.OpenAI()
openai_model = "gpt-4-1106-preview"

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = openai_model

if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = get_system_prompt()
    st.session_state.messages.append(get_message("system", system_prompt))
    companies_prompt = get_companies_prompt()
    st.session_state.messages.append(get_message("user", companies_prompt))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Please enter"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(get_message("user", prompt))
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            response_format={"type": "json_object"},
            messages=st.session_state.messages,
            stream=True,
        )
        for response in stream:
            full_response += response.choices[0].delta.content or ""
            message_placeholder.write(full_response + "▌")
        try:
            json_resp = json.loads(full_response)["data"]
            message_placeholder.json(json_resp)
            st.session_state.jsonresp = json_resp
        except:
            message_placeholder.write(full_response)
            st.session_state.jsonresp = full_response

    st.session_state.messages.append(get_message("assistant", full_response))

with st.expander("LLM Results"):
    if st.session_state.jsonresp is not None:
        try:
            st.dataframe(pd.DataFrame(st.session_state.jsonresp))
        except:
            st.write(st.session_state.jsonresp)
