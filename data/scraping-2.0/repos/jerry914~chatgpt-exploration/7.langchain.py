from langchain.agents import load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

agent_executor = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent_executor.invoke(
    {"input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"}
)
