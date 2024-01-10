import os
import random

import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX, SQL_PREFIX
from langchain.agents.agent_types import AgentType

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase


def general_config():
    st.set_page_config(
        page_title="LLMate",
        page_icon="imgs/llmate.png",
        layout='wide'
    )

    hide_menu = '''
    <style>
       MainMenu {visibility: hidden;}
       footer {visibility: hidden;}
    </style>
    '''
    st.markdown(hide_menu, unsafe_allow_html=True)
    
    # twitters = ['https://twitter.com/fpingham', 'https://twitter.com/petrallilucas', 'https://twitter.com/manuelsoria_']
    # random.shuffle(twitters)
    
    footer = f"<style> footer:after {{content:'Made with 🧉';\
    visibility: visible; display: block; position: relative; padding: 0px; top: -20px;}}</style>"
    st.markdown(footer, unsafe_allow_html=True)
