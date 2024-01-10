import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# First Agent
# ---------------------------
async_browser = create_async_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = browser_toolkit.get_tools()

llm = ChatOpenAI(temperature=0)  # Also works well with Anthropic models


# ZERO_SHOT_REACT_DESCRIPTION
# STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION <---------------------
# CONVERSATIONAL_REACT_DESCRIPTION
# CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# REACT_DOCSTORE
# SELF_ASK_WITH_SEARCH
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


async def main():
    response = await agent_chain.arun(input="Hi I'm Erica.")
    print(response)

    response = await agent_chain.arun(input="whats my name?")
    print(response)

    response = await agent_chain.arun(input="Don't need help really just chatting.")
    print(response)

    response = await agent_chain.arun(
        input="Browse to blog.langchain.dev and summarize the text, please."
    )
    print(response)

    response = await agent_chain.arun(input="What's the latest xkcd comic about?")
    print(response)


asyncio.run(main())
