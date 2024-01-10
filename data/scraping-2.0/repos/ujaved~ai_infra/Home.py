import streamlit as st
import os
import openai
from utils.terraform import download_terraform
from chatbot import VALIDATE_ERR_MSG, OpenAIChatbot, StreamlitStreamHandler, LLamaChatbot
from session import start_session, end_session

openai.api_key = os.getenv("OPENAI_API_KEY")
openai_model_id = os.getenv("OPENAI_MODEL_ID")
display_options = ['template', 'llm notes']


WELCOME_MSG = 'Welcome to InfraBot! Your bespoke AI-powered Terraform IaaS builder!'


def llm_selection_cb():
    # noop when no session has been started
    if len(st.session_state.sessions) == 0:
        return
    cur_session = st.session_state.sessions[-1]
    if st.session_state.llm == "codellama":
        cur_session.chatbot = LLamaChatbot(
            temperature=0, stream_handler_class=StreamlitStreamHandler)
    else:
        cur_session.chatbot = OpenAIChatbot(model_id=os.getenv(
            "OPENAI_MODEL_ID"), temperature=0, stream_handler_class=StreamlitStreamHandler)
    st.session_state.disable_llm_selection = True


def validate_template_cb():
    if len(st.session_state.sessions) == 0:
        return
    cur_session = st.session_state.sessions[-1]
    cur_session.static_validate = st.session_state.validate_template
    st.session_state.disable_validate_checkbox = True


def main():

    st.set_page_config(
        page_title="InfraBot", page_icon="👋", layout="wide"
    )
    st.sidebar.header('InfraBot')
    st.sidebar.write(
        'InfraBot is an AI-powered Terraform template builder. It generates and validates a terraform template for aws based on a conversation with you. It is built as an interface on top of ChatGPT.')
    if 'sessions' not in st.session_state:
        st.session_state.sessions = []

    if len(st.session_state.sessions) == 0:
        st.video(
            "https://ai-infra-demo-video-1.s3.amazonaws.com/streamlit-Home-2023-08-05-05-08.mp4")
    else:
        st.sidebar.video(
            "https://ai-infra-demo-video-1.s3.amazonaws.com/streamlit-Home-2023-08-05-05-08.mp4")

    with st.chat_message("assistant"):
        st.markdown(WELCOME_MSG)
        st.markdown('To start a new session, press the \'Start session\' button in the sidebar. When you are satisfied with your template you can press \'End session\' ')

    if 'disable_llm_selection' not in st.session_state:
        st.session_state.disable_llm_selection = False

    if 'disable_validate_checkbox' not in st.session_state:
        st.session_state.disable_validate_checkbox = False

    st.sidebar.button('Start Session', key='start_session',
                      type="primary", on_click=start_session)
    st.sidebar.radio("llm", ["gpt-4", "codellama"],
                     key="llm", on_change=llm_selection_cb, disabled=st.session_state.disable_llm_selection)
    st.sidebar.checkbox("validate template (static)", value=False,
                        key="validate_template", on_change=validate_template_cb, disabled=st.session_state.disable_validate_checkbox)

    st.sidebar.button('End Session', key=end_session, on_click=end_session)

    download_terraform(os.getenv("TERRAFORM_VERSION"))

    session_tab_names = ["session " + str(s.id)
                         for s in st.session_state.sessions]
    session_tab_names.reverse()
    session_tabs = []
    if len(session_tab_names) > 0:
        session_tabs = st.tabs(session_tab_names)

    for i, t in enumerate(reversed(session_tabs)):
        st.session_state.sessions[i].tab = t
        st.session_state.sessions[i].render()


if __name__ == "__main__":
    main()
