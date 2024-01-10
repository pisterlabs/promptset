"""
This is an agent to interact with a Pandas DataFrame.
This agent calls the Python agent under the hood, which executes LLM generated Python code.
"""


import streamlit as st
from langchain.agents import AgentType, create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]


def df_agent(df):
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
    )
    return agent
