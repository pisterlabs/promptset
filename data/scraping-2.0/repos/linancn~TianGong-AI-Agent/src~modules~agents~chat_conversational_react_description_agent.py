"""
This agent is optimized for conversation, which is ideal in a conversational setting where the agent to be able to chat with the user.
"""


import os

import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from ..tools.tools import (
    calculation_tool,
    innovation_assessment_tool,
    search_arxiv_tool,
    search_internet_tool,
    search_uploaded_docs_tool,
    search_vector_database_tool,
    search_wiki_tool,
)

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]


def agent_memory():
    llm_chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        streaming=False,
        verbose=False,
        callbacks=[],
    )
    memory = ConversationSummaryBufferMemory(
        llm=llm_chat,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=384,
    )
    return memory


def main_agent(memory=None):
    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
    )
    tools = [
        search_vector_database_tool,
        search_internet_tool,
        search_arxiv_tool,
        search_wiki_tool,
        search_uploaded_docs_tool,
        calculation_tool(),
        innovation_assessment_tool,
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=langchain_verbose,
    )
    return agent
