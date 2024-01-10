import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import streamlit as streamlit
from streamlit_chat import message

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

streamlit.set_page_config(
    page_title='Your Custom Assistance',
    page_icon='🤖'
)
st.subheader('Your Custom ChatGPT 🤖')
chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
if 'message' not in st.session_state:
    st.session_state.message = []

with st.sidebar:
    system_message = st.text_input(label='System role')
    user_prompt = st.text_input(label='Send a message')
    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.message):
            st.session_state.message.append(
                SystemMessage(content=system_message)
            )

        # st.write(st.session_state.message)

    if user_prompt:
        st.session_state.message.append(
            HumanMessage(content=user_prompt)
        )
        with st.spinner('Working on your request ...'):
            response = chat(st.session_state.message)

        st.session_state.message.append(AIMessage(content=response.content))

# st.session_state.message

# message('This is chatgpt', is_user=False)
# message('This is the user', is_user=True)

if len(st.session_state.message) >= 1:
    if not isinstance(st.session_state.message[0], SystemMessage):
        st.session_state.message.insert(0, SystemMessage(content='Your are a helpful assistant.'))

for i,msg in enumerate(st.session_state.message[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=f'{i} + 😎')
    else:
        message(msg.content, is_user=False, key=f'{i} + 🤖')
