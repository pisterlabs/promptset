from langchain import agents
from langchain.llms.base import BaseLanguageModel


def make_tools(llm: BaseLanguageModel):
    # add human input tool
    tools = agents.load_tools(["human"], llm)

    # append tools here
    tools += []
    return tools
