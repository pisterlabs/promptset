import os
os.environ["OPENAI_API_KEY"] = "ENTER OPENAI API KEY HERE"
os.environ["SERPAPI_API_KEY"] = "ENTER SERPAPI KEY HERE"

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=1)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("what should I do to become an expert on Large language models and GPT?")

