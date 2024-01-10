# Databricks notebook source
"""LangChain: Agents
Outline:
Using built in LangChain tools: DuckDuckGo search and Wikipedia
Defining your own tools"""

# COMMAND ----------

import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

# COMMAND ----------

"""Built-in LangChain tools"""

# COMMAND ----------

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

# COMMAND ----------

llm = ChatOpenAI(temperature=0, model=llm_model)

# COMMAND ----------

tools = load_tools(["llm-math","wikipedia"], llm=llm)

# COMMAND ----------

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

# COMMAND ----------

agent("What is the 25% of 300?")

# COMMAND ----------

"""Wikipedia example"""

# COMMAND ----------

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 

# COMMAND ----------

"""Python Agent"""

# COMMAND ----------

agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

# COMMAND ----------

customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

# COMMAND ----------

agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 

# COMMAND ----------

import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False

# COMMAND ----------

"""Define your own tool"""

# COMMAND ----------

from langchain.agents import tool
from datetime import date

# COMMAND ----------

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

# COMMAND ----------

agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

# COMMAND ----------

try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


