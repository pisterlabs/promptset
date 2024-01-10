
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

import os

# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["SERPAPI_API_KEY"] = "7c837ddfd17623f5ac9b3717c8e984f155759e47ec1fc87e6aea0a174a1f4c3d"
#
# llm = OpenAI(temperature=0)
# tools = load_tools(["serpapi", "llm-math"], llm=llm)
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")




