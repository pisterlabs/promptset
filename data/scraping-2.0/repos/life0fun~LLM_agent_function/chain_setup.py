import os
import getpass
from langchain.globals import set_debug
from langchain.utilities import SerpAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities import PubMedAPIWrapper
from langchain import ArxivAPIWrapper, LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.tools import StructuredTool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

from langchain.prompts.chat import MessagesPlaceholder
import tools_wrappers
from typing import Tuple, Dict

# Globals
# os.environ["SERPAPI_API_KEY"] = getpass.getpass()

class Config():
    """
    Contains the configuration of the LLM.
    """    
    model = 'gpt-3.5-turbo-16k-0613'
    llm = ChatOpenAI(temperature=0, model=model, openai_api_key=os.getenv("OPENAI_KEY"))


def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    """
    Sets up memory for the open ai functions agent.
    :return a tuple with the agent keyword pairs and the conversation memory.
    """
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    return agent_kwargs, memory

def setup_agent() -> AgentExecutor:
    """
    Sets up the tools for a function based chain.
    We have here the following tools:
    - wikipedia
    - google
    - calculator
    - arxiv
    - events (a custom tool)
    - pubmed
    """
    cfg = Config()
    google_search = SerpAPIWrapper()
    wikipedia = WikipediaAPIWrapper()
    pubmed = PubMedAPIWrapper()
    events = tools_wrappers.EventsAPIWrapper()
    events.doc_content_chars_max=5000
    llm_math_chain = LLMMathChain.from_llm(llm=cfg.llm, verbose=False)
    arxiv = ArxivAPIWrapper()
    tools = [
        Tool(
            name = "GoogleSearch",
            func=google_search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="useful when you need an answer about encyclopedic general knowledge"
        ),
        Tool(
            name="Arxiv",
            func=arxiv.run,
            description="useful when you need an answer about encyclopedic general knowledge"
        ),
        StructuredTool.from_function(
            func=events.run,
            name="Events",
            description="useful when you need an answer about meditation related events in the united kingdom"
        ),
        StructuredTool.from_function(
            func=pubmed.run, 
            name='PubMed',
            description='Useful tool for querying medical publications'
        )
    ]
    agent_kwargs, memory = setup_memory()

    return initialize_agent(
        tools, 
        cfg.llm, 
        agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
        verbose=True, 
        agent_kwargs=agent_kwargs,
        memory=memory
    )