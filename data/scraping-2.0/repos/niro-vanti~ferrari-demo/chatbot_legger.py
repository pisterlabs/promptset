import os
from dotenv import load_dotenv
from io import BytesIO
from io import StringIO
import sys
import re
from langchain.agents import create_csv_agent
from src.modules.history import ChatHistory
from src.modules.layout import Layout
from src.modules.utils import Utilities
from src.modules.sidebar import Sidebar
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from json2table import convert


# To be able to update the changes made to modules in localhost,
# you can press the "r" key on the localhost page to refresh and reflect the changes made to the module files.
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]


history_module = reload_module('src.modules.history')
layout_module = reload_module('src.modules.layout')
utils_module = reload_module('src.modules.utils')
sidebar_module = reload_module('src.modules.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar


def init():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="💬", page_title="ChatBot-Legger")


def main():
    init()
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()
    sidebar.show_logo('assets/Images/colleen-logo.png')

    layout.show_header_txt()
    user_api_key = utils.load_api_key()

    if not user_api_key:
        layout.show_api_key_missing()
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key
        # uploaded_file = utils.handle_upload_txt()
        uploaded_file = utils.handle_upload_ledger()

        if uploaded_file:
            history = ChatHistory()
            sidebar.show_options()

            uploaded_file_content = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = uploaded_file_content.read()

            # st.write(string_data)

            try:
                chatbot = utils.setup_chatbot_ledger(
                    uploaded_file, st.session_state["model"], st.session_state["temperature"]
                )
                # st.write(chatbot)
                st.session_state["chatbot"] = chatbot
                st.session_state['agent'] = chatbot

                if st.session_state["ready"]:
                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        is_ready, user_input = layout.prompt_form()

                        history.initialize(uploaded_file)
                        if st.session_state["reset_chat"]:
                            history.reset(uploaded_file)

                        if is_ready:
                            history.append("user", user_input)
                            output = st.session_state["chatbot"].csv_agent(user_input)
                            # st.text(sys.stdout)
                            # st.text(StringIO.getvalue())
                            old_stdout = sys.stdout
                            sys.stdout = captured_output = StringIO()
                            sys.stdout = old_stdout
                            thoughts = captured_output.getvalue()
                            # st.text(old_stdout)
                            # st.text(captured_output)
                            st.text('thoughts: '+thoughts)

                            history.append("assistant", output)

                    history.generate_messages(response_container)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    sidebar.about()


if __name__ == "__main__":
    main()
