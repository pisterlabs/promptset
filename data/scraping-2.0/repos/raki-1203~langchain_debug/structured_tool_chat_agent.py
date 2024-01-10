import os

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['LANGCHAIN_TRACING'] = 'true'  # If you want to trace the execution of the program, set to "true"


if __name__ == '__main__':
    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = browser_toolkit.get_tools()

    for tool in tools:
        print(tool)

    llm = ChatOpenAI(temperature=0)  # Also works well with Anthropic models
    agent_chain = initialize_agent(tools, llm,
                                   agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                   verbose=True)

    # await error 발생
    response = agent_chain.run(input="Hi I'm Erica.")
    print(response)

    response = agent_chain.run(input="Don't need help really just chatting.")
    print(response)

    response = agent_chain.run(input="Browse to blog.langchain.dev and summarize the text, please.")
    print(response)

    response = agent_chain.run(input="What's the latest xkcd comic about?")
    print(response)

    response = agent_chain.run(input="Browse to blog.langchain.dev and summarize the text, please.")
    print(response)
