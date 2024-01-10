import os
import asyncio
import nest_asyncio
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_async_playwright_browser

# Load environment variables from .env file
load_dotenv()

# Access the API keys from the environment variables
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Set the tracing environment variable
os.environ["LANGCHAIN_TRACING"] = "true" # If you want to trace the execution of the program, set to "true"

#from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,# A synchronous browser is available, though it isn't compatible with jupyter.
)


async_browser = create_async_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = browser_toolkit.get_tools()

llm = ChatOpenAI(temperature=0) # Also works well with Anthropic models
agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

async def run_agent():
    response = await agent_chain.arun(input="impedance control Methods for force feedback")
    print(response)

nest_asyncio.apply() # Enable asyncio event loop nesting
loop = asyncio.get_event_loop()
loop.run_until_complete(run_agent())
