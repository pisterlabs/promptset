import os

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


if __name__ == '__main__':
    llm = OpenAI(temperature=0)

    search = SerpAPIWrapper()

    tools = [
        Tool(
            name='Intermediate Answer',  # AgentType.SELF_ASK_WITH_SEARCH 사용시 name 을 `Intermediate Answer` 로 사용해야 함
            func=search.run,
            description="useful for when you need to ask with search",
        ),
    ]

    self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)

    print(self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?"))

