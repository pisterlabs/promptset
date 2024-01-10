from langchain import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools

from botchan import settings

DEFAULT_LOAD_TOOLS = [
    # math
    "llm-math",  # mathtool via llm
    # knowledge
    # "wolfram-alpha",
    "wikipedia",
    "arxiv",
    # news
    "news-api",
    # search
    "serpapi",  # search via serp api
]


def create_default_agent(tools: list = None):
    llm = OpenAI(temperature=0)
    if tools is None:  # load default tools
        tools = load_tools(
            settings.MARK_LOAD_TOOLS, llm=llm, news_api_key=settings.NEWS_API_KEY
        )

    mrkl = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return mrkl
