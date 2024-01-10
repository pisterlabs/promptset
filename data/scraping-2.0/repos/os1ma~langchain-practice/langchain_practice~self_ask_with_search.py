from langchain import OpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from util import initialize

initialize()

llm = OpenAI(temperature=0)
search = DuckDuckGoSearchAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH
)
self_ask_with_search.run(
    "What is the hometown of the reigning men's U.S. Open champion?"
)
