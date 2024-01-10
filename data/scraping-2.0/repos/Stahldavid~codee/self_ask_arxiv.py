from langchain import OpenAI, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
from langchain.utilities import ArxivAPIWrapper
import os



# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

llm = ChatOpenAI(temperature=0)
arxiv = ArxivAPIWrapper()
tools = [
    Tool(
        name="Arxiv",
        func=arxiv.run,
        description="useful for searching academic papers. It's input is a specific query, not too vague."
    )
]

with_search = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
with_search.run("Best method of variable impedance control for force feedback and haptics.")