import os
import sys

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)

import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import FileCallbackHandler
from app_streamlit.retrieve import build_chain, run_chain
from loguru import logger

load_dotenv()

EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL')
EMBEDDINGS_PATH = os.environ.get('EMBEDDINGS_PATH')
MAX_HISTORY_LENGTH = int(os.environ.get('MAX_HISTORY_LENGTH'))
REPHRASED_TOKEN = os.environ.get('REPHRASED_TOKEN') # This helps streamlit to ignore the response from the API used to rephrase the question based on history
LOG_FILE_PATH = os.path.join(MAIN_DIR, os.environ.get('LOG_FILE_PATH'))

logger.add(LOG_FILE_PATH, colorize=True, enqueue=True)
log_handler = FileCallbackHandler(LOG_FILE_PATH)

st.set_page_config(page_title="AIAssistant-Aznor", page_icon="🧑‍💼")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", rephrased_token=REPHRASED_TOKEN):
        self.container=container
        self.text=initial_text
        self.is_rephrased=None
        self.rephrased_token=REPHRASED_TOKEN

    def on_llm_new_token(self, token, **kwargs):
        if self.rephrased_token not in token:
            self.text+=token
            self.container.markdown(self.text + "▌")


def render_app():
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
        <style>
        button {
            height: 30px !important;
            width: 150px !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.subheader("Hello, I am an AI Assistant. \n\n I am here to share more about Aznor and how he can be an asset to your organization. \n\n Ask me anything in the chat box below.")

    if "chat_dialogue" not in st.session_state:
        st.session_state["chat_dialogue"] = []

    if "chat_dialogue_display" not in st.session_state:
        st.session_state["chat_dialogue_display"] = []

    def clear_history():
        st.session_state["chat_dialogue"] = []

    def clear_history_all():
        st.session_state["chat_dialogue"] = []
        st.session_state["chat_dialogue_display"] = []

    embedding_function = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDINGS_MODEL,
        cache_folder=EMBEDDINGS_PATH
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_dialogue_display:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.chat_dialogue) >= MAX_HISTORY_LENGTH:
        clear_history()

    if prompt := st.chat_input("Type your questions here..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display message from LLM
        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            stream_handler = StreamHandler(answer_placeholder)
            chain = build_chain(embedding_function, [stream_handler, log_handler])
            try:
                output = run_chain(chain, prompt, st.session_state.chat_dialogue)
                answer = output["answer"]
                logger.info(output)
            except Exception:
                output = {}
                answer = "I am sorry I am unable to respond to your question."
                logger.info(output)
                answer_placeholder.markdown(answer + "▌")
            if 'source_documents' in output:
                with st.expander("Documents Referenced"):
                    for _sd in output.get('source_documents'):
                        _sd_metadata = _sd.metadata
                        source = _sd_metadata.get("source")
                        st.text(f"Location: {source}")
            # Add user message to chat history and display
            st.session_state.chat_dialogue.append({"role": "user", "content": prompt})
            st.session_state.chat_dialogue_display.append({"role": "user", "content": prompt})
            # Add assistant response to chat history and display
            st.session_state.chat_dialogue.append({"role": "assistant", "content": answer})
            st.session_state.chat_dialogue_display.append({"role": "assistant", "content": answer})
        col1, col2 = st.columns([10, 4])
        with col1:
            pass
        with col2:
            st.button("Clear History", use_container_width=True, on_click=clear_history_all)

render_app()




