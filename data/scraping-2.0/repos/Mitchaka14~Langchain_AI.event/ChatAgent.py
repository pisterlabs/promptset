# main.py

import os
import json
from langchain import LLMMathChain, SerpAPIWrapper, OpenAI, LLMChain
from langchain.agents import (
    AgentType,
    initialize_agent,
    Tool,
    ZeroShotAgent,
    AgentExecutor,
)
from langchain.chat_models import ChatOpenAI
from tools.my_tools import DataTool, SQLAgentTool
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain


def event_handler(event_type, message):
    if event_type == "input_required":
        return input(message)


import streamlit as st
from dotenv import load_dotenv

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    os.environ["serpapi_api_key"] = st.secrets["SERPAPI_API_KEY"]
except KeyError:
    load_dotenv()
    os.environ["serpapi_api_key"] = os.getenv("SERPAPI_API_KEY")
# _______________________________________________________________________


# Initialize the LLM to use for the agent.

search = SerpAPIWrapper()

# Construct the agent.
tools = [
    DataTool(),
    SQLAgentTool(),
    # InteractiveTool(event_handler=event_handler),
    # FeedbackTool(),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to ask with search......dont end conversation unless, the customer is satisfied",
    ),
]
# ____________________________________________________________________________________________________________________________________________________

prefix = """dont end conversation unless, the customer is satisfied
You are a customer service agent named Jack. You are friendly but firm and don't let customers make unnecessary changes to the database.
If they want to make changes, you have to confirm with their name if it's them.
You don't end the conversation until the user is either satisfied or wants to leave.
always confirm before ending the conversation, follow proper etiquette
You have access to the following tools:"""
suffix = """Begin!   

{chat_history}
CustomerInput: {input}
dont end conversation unless, the customer is satisfied"
{agent_scratchpad}
dont end conversation unless, the customer is satisfied"
"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")
# llm = ChatOpenAI(temperature=0, model="gpt-4-0613", prompt=prompt)
llm_chain = LLMChain(llm=ChatOpenAI(temperature=0, model="gpt-4-0613"), prompt=prompt)
# ____________________________________________________________________________________________________________________________________________________


agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools,
    verbose=True,
)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, max_iterations=1000
)


# Start the conversation
initial_input = "Start conversation with customer, dont end conversation till customer is satisfied "
full_prompt = initial_input

json_output = agent_chain.run(input=full_prompt)

# Print the JSON output. Depending on the structure of your JSON, you might want to parse it into a Pydantic model, as I showed in the previous examples.
print(json_output)
