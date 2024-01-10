import os
import openai
import weaviate
import streamlit as st

from globals import (DEFAULT_MODEL_CONFIG, DB_TABLE)

# MAIN APP STATE ----------------------------------------------------------------

state = st.session_state

# Initial state builder
def build_initial_state():
    openai_api_key = None
    if st.secrets.get('OPENAI_API_KEY', None):
        print('settings', 'OPENAI_API_KEY found')
        openai_api_key = st.secrets['OPENAI_API_KEY']
    else:
        print('settings OPENAI_API_KEY not found!')
        # Try get OpenAI api key from os env
        # (this is the workaround for using Streamlit in Heroku)
        if os.environ.get('OPENAI_API_KEY', None):
            print('os.environ', 'OPENAI_API_KEY found')
            openai_api_key = os.environ['OPENAI_API_KEY']
            openai.api_key = os.getenv("OPENAI_API_KEY")

    print('openai_api_key', 'sk_...' + openai_api_key[-5:], '\n') if openai_api_key else print('openai_api_key', 'NULL', '\n')

    weaviate_api_key = st.secrets.get('WEAVIATE_API_KEY', None)
    print('weaviate_api_key',  weaviate_api_key[:5] + '...' + weaviate_api_key[-5:], '\n') if weaviate_api_key else print('weaviate_api_key', 'NULL', '\n')

    WEAVIATE_URL = st.secrets.get('WEAVIATE_URL', None)
    
    initial_state = {
        # MAIN APP STATE
        'openai_api_key': openai_api_key,
        'weaviate_api_key': weaviate_api_key,
        'WEAVIATE_URL': WEAVIATE_URL,
        'menu_choice': 0,
    
        # DATA PAGE STATE
        'limit': 3,
        'query': '',
        'intermediate_steps': True,
        'db_table': DB_TABLE,
        'generated': [],
        'past': [],
        'questions': [],
    
        # KNOWLEDGE GRAPH PAGE STATE
        'user_input': '',

        # MODEL STATE
        'chat_model': DEFAULT_MODEL_CONFIG['chat_model'],
        'completions_model': DEFAULT_MODEL_CONFIG['completions_model'],
        'temperature': DEFAULT_MODEL_CONFIG['temperature'],
        'top_p': DEFAULT_MODEL_CONFIG['top_p'],
        'max_tokens': DEFAULT_MODEL_CONFIG['max_tokens'],

        'estimated_cost_reset': 0,
        'estimated_cost_data': 0,
        'estimated_cost_doc': 0,
        'estimated_cost_graph': 0,
        'cumulative_cost': 0,
    }
    
    return initial_state

# State initializer
def init_app_state():
    initial_state = build_initial_state()
    for k, v in initial_state.items():
        if not state.get(k, None):
            setattr(state, k, v)

# State resetter
def reset_app_state():
    initial_state = build_initial_state()
    for k, v in initial_state.items():
        setattr(state, k, v)

# STATE CALLBACK ----------------------------------------------------

# generic callback to set state
def _set_state_cb(**kwargs):
    for state_key, widget_key in kwargs.items():
        val = state.get(widget_key, None)
        if val is not None or val == "":
            setattr(state, state_key, state[widget_key])
