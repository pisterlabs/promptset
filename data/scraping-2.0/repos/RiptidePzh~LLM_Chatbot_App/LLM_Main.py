import time

import streamlit as st
from friend_replica.format_chat import ChatConfig, format_chat_history, split_chat_data
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *
from langchain.llms import GPT4All

# from models.model_cn import ChatGLM



### Side Bar Module ###
with st.sidebar:
    openai_api_key = st.text_input("Comma API Key", key="chatbot_api_key", type="password")
    "[Get a Comma API key](https://github.com/roxie-zhang/friend_replica)"
    "[View the source code](https://github.com/roxie-zhang/friend_replica)"

### Header Module ###
st.title("Comma Language ChatBot")
st.caption("🚀 A chatbot powered by BigDL on-device LLM | *FDU Comma Team Ver-1.1*")
# st.markdown('---')

### Config Model ###
st.subheader('LLM Replica Configuration')
st.caption("Before Start using those amazing features, you need to first load your chat "
           "history to create a LLM agent. Make sure you fill in the configuration accordingly.")

config_container = st.container()
config_form = config_container.form('Config Model')
col1, col2, col3 = config_form.columns(3)
st.session_state.my_name = col1.text_input('Your Name')
st.session_state.friend_name = col2.text_input('Frined Name')
st.session_state.language = col3.selectbox('Select Language',['chinese','english'])
button = config_form.form_submit_button('Config')

st.session_state.current_chat_replica = []
st.session_state.current_chat_archieve = []
st.session_state.continue_chat = False
st.session_state.current_idx = -1

### Configuration ###
def chat_config():
    my_bar = st.progress(0, "Operation in progress. Please wait.")
    time.sleep(1)
    my_bar.progress(10, text="Operation in progress. Please wait.")
    time.sleep(1)
    chat_config = ChatConfig(
        my_name=st.session_state.my_name,
        friend_name=st.session_state.friend_name,
        language=st.session_state.language,
    )
    my_bar.progress(30, text="Initializing Model...")
    st.session_state.chat_with_friend = Chat(device='cpu', chat_config=chat_config)
    my_bar.progress(75, text="Vectorization...")
    time.sleep(1)
    st.session_state.chat_blocks = split_chat_data(st.session_state.chat_with_friend.chat_data)
    #st.write([len(c) for c in st.session_state.chat_blocks])
    my_bar.progress(100, text="Configuration Finished")

if button:
    try:
       chat_config()
       st.success('Configuration Success!')
    except Exception as e:
        st.warning('Error During Configuration')
        st.warning(e)

### End of the page ###
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown('---')
st.markdown('> *This demo version is made by Zihan for Learning use only*')