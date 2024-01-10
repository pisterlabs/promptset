import streamlit as st
import random
import time
from openai import OpenAI

st.title("Echo Bot")

# st.info("API_KEY: {}".format(st.secrets["OPENAI_API_KEY"]))


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-1106"

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        


prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response:str = ""

        chat_response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role":m["role"],"content":m["content"]} for m in st.session_state["messages"]],
            stream=True
        )
        

    for response in chat_response:
        if(response.choices[0].delta.content):
            full_response += response.choices[0].delta.content

        message_placeholder.markdown(full_response)
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role":"assistant", "content":full_response})
    








# for response in chat_response:
#     full_response += (response.choices[0].delta.content or "")
#     st.markdown(full_response)

    #     for response in chat_response:
    #         full_response += (response.choices[0].delta.content or "")
    #         message_placeholder.markdown(full_response  + " ")
    #     message_placeholder.markdown(full_response)
    # st.session_state.messages.append({"role":"assistant", "content":full_response})    

