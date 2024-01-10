"""
This agent uses the ReAct framework to determine tools to be use to solve the prompts.
This is the most general purpose action agent.
"""


import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.python.tool import PythonREPLTool

from ..tools.search_arxiv_tool import SearchArxivTool
from ..tools.search_google_patents_tool import SearchGooglePatentsTool
from ..tools.search_internet_tool import SearchInternetTool
from ..tools.search_vectordb_tool import SearchVectordbTool
from ..tools.search_wikipedia_tool import SearchWikipediaTool
from ..tools.st_search_uploaded_docs_tool import STSearchUploadedDocsTool


def main_agent():
    llm_model = st.secrets["llm_model"]
    langchain_verbose = st.secrets["langchain_verbose"]

    llm_chat = ChatOpenAI(
        model=llm_model,
        temperature=0,
        streaming=True,
        verbose=langchain_verbose,
        callbacks=[],
    )
    tools = [
        SearchInternetTool(),
        SearchVectordbTool(),
        SearchArxivTool(),
        SearchWikipediaTool(),
        SearchGooglePatentsTool(),
        STSearchUploadedDocsTool(),
        PythonREPLTool(),
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm_chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=langchain_verbose,
    )
    return agent
