from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.agents import initialize_agent, Tool

def get_tool():
    tool = Tool(
        name="Search",
        func=SerpAPIWrapper().run,
        description="Useful when you need to answer questions about current events. You should ask targeted questions.",
    )
    return tool