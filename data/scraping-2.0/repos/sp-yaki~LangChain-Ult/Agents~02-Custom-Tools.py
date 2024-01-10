import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

from langchain.agents import tool

@tool
def coolest_guy(text: str) -> str:
    '''Returns the name of the coolest guy in the universe'''
    return "Jose Portilla"

tools = load_tools(["wikipedia","llm-math"], llm=llm) 
tools = tools +[coolest_guy]

agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True)
agent.run("Whos is the coolest guy in the universe?")


# @tool
# def some_api_call(text: str) -> str:
#     '''Can now connect your Agents to any tool via an API call, get creative here!'''
#     return api_result

from datetime import datetime
# DOC STRINGS SHOULD BE VERY DESCRIPTIVE
# IT IS WHAT THE LLM READS TO DECIDE TO USE THE TOOL!
@tool
def get_time(text: str) -> str:
    '''Returns the current time. Use this for any questions
    regarding the current time. Input is an empty string and
    the current time is returned in a string format. Only use this function
    for the current time. Other time related questions should use another tool'''
    return str(datetime.now())

agent = initialize_agent(tools+[get_time], 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True)

agent("What time did Pearl Harbor happen at?")
agent("What time is it?")