from llama_index import GPTSimpleVectorIndex 
from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish
import re
import os
from langchain.agents import initialize_agent, Tool, ZeroShotAgent, AgentExecutor
from typing import Any, List, Optional, Tuple, Union
from berri_ai.QAAgent import QAAgent

class ComplexInformationQA():
  """Base class for Complex Information QA Agent Class"""  

  def __init__(self, openai_api_key, index = None, prompt = None, functions = None, descriptions = None):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    self.index = index 
    if len(functions) > 0:
      if len(functions) != len(descriptions):
        raise ValueError("The number of functions does not match the number of descriptions!")
      self.functions = functions 
      self.descriptions = descriptions 
      tools = self.define_tools()
    else: 
      tools = [
        Tool(
            name = "QueryingDB",
            func=function,
            description="This function takes a query string as input and returns the most relevant answer from the documentation as output"
        )]

    PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""

    SUFFIX = """Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    self.prompt = ZeroShotAgent.create_prompt(
          tools, 
          prefix=PREFIX, 
          suffix=SUFFIX, 
          input_variables=["input", "agent_scratchpad"])
    self.tools = tools
    self.llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=self.prompt)
 
  
  def define_tools(self): 
    agent_tools = []
    for idx, function in enumerate(self.functions): 
      # set tool name
      tool_name = self.extract_function_names(function)
      # set tool func
      tool_func  = function
      # initialize tool for the agent
      agent_tool = Tool(name=tool_name, func=tool_func, description=self.descriptions[idx])
      # add to list of tools
      agent_tools.append(agent_tool)
    return agent_tools

  # define function that takes in a list of functions
  def extract_function_names(self, function): 
    # extract the name from the function
    name = function.__name__ 
    # append the name to the list
    return name

  def querying_db(self, query: str):
    response = self.index.query(query)
    response = (response.response, response.source_nodes[0].source_text)
    return response
  
  def run(self, query_string: str):
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=self.prompt)
    agent2 = QAAgent(llm_chain=self.llm_chain, tools=self.tools)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent2, tools=self.tools, verbose=True, return_intermediate_steps=True)
    answer = agent_executor({"input":query_string})
    return answer