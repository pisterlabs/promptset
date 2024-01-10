import os
import openai
import streamlit as st


def set_openai_keys(api_key, api_base):
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_API_BASE'] = api_base
    openai.api_key = api_key
    openai.api_base = api_base


def init_session_state():
    if "server_api_key" not in st.session_state:
        st.session_state.server_api_key = "EMPTY"
    if "server_api_base" not in st.session_state:
        st.session_state.server_api_base = "https://u31193-9a50-96959e64.neimeng.seetacloud.com:6443/v1" #"https://api.openai.com/v1"    
    if "generate_params" not in st.session_state:
        st.session_state.generate_params = {'max_tokens':1024, 'temperature':0.9, 'top_p':0.6}
    if "tool_names" not in st.session_state:
        st.session_state.tool_names = ["Wikipedia", "Browse Website", "LLM Code"]
    if "agent_names" not in st.session_state:
        st.session_state.agent_names = ["Web Copilot", "Code Copilot", "Shell Copilot"]
    if "chat_model_name" not in st.session_state:
        st.session_state.chat_model_name = "gpt-3.5-turbo"
    if "code_model_name" not in st.session_state:
        st.session_state.code_model_name = "code-llama"
    if "completion_model_name" not in st.session_state:
        st.session_state.completion_model_name = "text-davinci-003"
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = "text-embedding-ada-002"
    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = "openchat_3.5"
    if "system_message" not in st.session_state:
        st.session_state.system_message = "You are Vic, an AI assistant that follows instruction extremely well. Help as much as you can."
    if "translation" not in st.session_state:
        st.session_state.translation = ""
    if "translation_lang" not in st.session_state:
        st.session_state.translation_lang = "中文"
    if "wiki_lang" not in st.session_state:
        st.session_state.wiki_lang = "en"
    if "amazing_note" not in st.session_state:
        st.session_state.amazing_note = ""
    if "env_key_data" not in st.session_state:
        st.session_state.env_key_data = {}
        for env_name, key in st.session_state.env_key_data.items():
            os.environ[env_name] = key
