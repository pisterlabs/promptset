import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")

tools = load_tools(["llm-math"], llm=llm)
agent_chain = initialize_agent(tools, 
                               llm,
                               agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                               verbose=True, 
                               memory=memory)
agent_chain.run(input="What are some good thai food recipes?")
print("-------------------------------------------")
agent_chain.run("Which one of those dishes is the spiciest?")
print("-------------------------------------------")
agent_chain.run("Give me a grocery shopping list to make that dish")