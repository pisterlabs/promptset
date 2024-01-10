import streamlit as st
import openai
import pandas as pd
#from Capstone import key_config
import assistantsetup as adv
import time


# case1data = pd.read_csv("StreamlitApp/data/patient1.csv")
# Assistant1 = adv.Assistant("patient1.csv")

st.title("AI-betes Advisor")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

Assistant1 = adv.Assistant("patient1.csv")
if Assistant1.file == None:
    Assistant1.getFile()

if Assistant1.thread == None:
    Assistant1.generateThread()

if Assistant1.run == None:
    Assistant1.generateRun()

if prompt := st.chat_input("What is on your mind?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    Assistant1.wait_on_run()
    assistant_response = Assistant1.retrieveMessage()
    # if assistant_response.
    message_placeholder.markdown(assistant_response.value)
    for annotation in assistant_response.annotations:
        message_placeholder.markdown(assistant_response.annotation)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": assistant_response.value})
for annotation in assistant_response.annotations:
    st.session_state.messages.append({"role": "assistant", "content": assistant_response.annotation})
