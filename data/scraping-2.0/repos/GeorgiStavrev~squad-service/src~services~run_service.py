
import os
from typing import List
from types import ModuleType
from inspect import getmembers, isfunction

import openai

from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import StructuredTool

openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


def run(module: ModuleType, query: str) -> str:
    tools = [StructuredTool.from_function(fn)
             for _, fn in getmembers(module, isfunction) if hasattr(fn, "__isskill")]
    agent = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    return agent.run(query)
