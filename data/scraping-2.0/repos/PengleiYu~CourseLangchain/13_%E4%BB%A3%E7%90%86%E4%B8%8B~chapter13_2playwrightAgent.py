# 结构化代理使用playwright工具
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_async_playwright_browser

browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
tools = toolkit.get_tools()

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI, ChatAnthropic

# 需要多试几次，经常出现第一步就结束的情况
llm = ChatOpenAI(temperature=0.5)
agent = initialize_agent(tools=tools,
                         llm=llm,
                         agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)


async def main():
    response = await agent.arun("What are the headers on python.langchain.com?")
    print(response)


import asyncio

loop = asyncio.get_event_loop()
loop.run_until_complete(future=main())
