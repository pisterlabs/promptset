import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# First Agent
# ---------------------------
llm = OpenAI(temperature=0)


# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# ZERO_SHOT_REACT_DESCRIPTION <---------------------
# STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# CONVERSATIONAL_REACT_DESCRIPTION
# CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# REACT_DOCSTORE
# SELF_ASK_WITH_SEARCH


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Now let's test it out!
agent.run(
    "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?"
)
