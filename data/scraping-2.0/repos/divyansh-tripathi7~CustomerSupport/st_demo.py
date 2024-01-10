import streamlit as st
from main import chain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

st.header(" Walkout wear assistant")
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant.")
    ]

with st.sidebar:
    user_input = st.text_input("Your message: ", key="user_input")


    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        query = user_input
        with st.spinner("Thinking..."):
            response = chain({"question": query})
            print(response['result'])
            st.write(response['result'])
            st.session_state.messages.append(
                AIMessage(content=response['result']))
    # display message history
messages = st.session_state.get('messages', [])
for i, msg in enumerate(messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=str(i) + '_user')
    else:
        message(msg.content, is_user=False, key=str(i) + '_ai')
