from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

## Add tools here

### Import the og templates that are needed to run this


####
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    def format(self, **kwargs) -> str:
        ## Get the intermaediate steps (agentAction, observarion tuples) 
        # Format them in a particular way
        intermediate_steps = kwargs["intermediate_steps"]
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action}\n"
            thoughts += f"Observation: {observation}\n"
        kwargs["agent_scratchpad"]

## Format the output of the agent

### Set up LLM

### Define stop conditions

### Set up the agent

### Use the agent


### Adding memory to the agent

### More detail found on https://python.langchain.com/en/latest/modules/agents/agents/custom_llm_agent.html