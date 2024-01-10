import os
from langchain.prompts import PromptTemplate
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


os.environ['OPENAI_API_KEY'] = ''
llm = OpenAI(temperature = 0.9)
tools = [PythonREPLTool()]
topic = input("What tool do you want to make?: " )
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
prompt =  '''Use python code based off of this template 
from langchain.tools import StructuredTool
def multiplier(a: float, b: float) -> float:
    """Multiply the provided floats."""
    return a * b
multtool = StructuredTool.from_function(multiplier)
To create a tool that {}, and in your final answer, paste only the function and the assignment DO NOT IMPORT ANYTHING,
remember to format the types correctly too, DO NOT NAME IT multtool'''.format(topic)

respones  = agent.run(prompt)
print(respones)
yn = input("Do you want to append this?: ")
if yn in ['y', 'Y','yes']:
    content_to_append = '\n'+respones
    with open('tools.py', 'a') as file:
        file.write(content_to_append)
    print('appended')
else:
    print('append tool cancelled')