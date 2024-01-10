import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

os.getenv("OPENAI_API_KEY")
os.getenv("AMADEUS_CLIENT_ID")
os.getenv("AMADEUS_CLIENT_SECRET")

from langchain.agents.agent_toolkits.amadeus.toolkit import AmadeusToolkit

toolkit = AmadeusToolkit()
tools = toolkit.get_tools()
print(tools)
print(", ".join([t.name for t in tools]))

from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    verbose=False,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

a = agent.run("What is the departure time of the cheapest flight on December 23, 2023 leaving Dallas, Texas before noon to Lincoln, Nebraska?")
print(a)