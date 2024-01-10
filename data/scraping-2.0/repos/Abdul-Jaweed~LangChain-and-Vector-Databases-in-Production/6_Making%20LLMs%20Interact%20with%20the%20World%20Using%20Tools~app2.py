from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import OpenAI

import os
from dotenv import load_dotenv

api_key = os.get_env("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=api_key,
    temperature=0
)


tools = load_tools(
    ["requests_all"],
    llm=llm
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run("Get the list of users at https://644696c1ee791e1e2903b0bb.mockapi.io/user and tell me the total number of users")