import tools
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


import ast
import importlib
import sys

def gather_tools_from_file(module_name='tools'):
    # Step 1: Parse the module to get the tool variable names
    with open(f"{module_name}.py", 'r') as file:
        content = file.read()

    parsed_content = ast.parse(content)

    tool_variable_names = []
    for node in ast.walk(parsed_content):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if 'tool' in var_name:
                    tool_variable_names.append(var_name)

    # Step 2: Dynamically import the module and extract the tools
    if module_name in sys.modules:
        # If already imported, reload it
        imported_module = importlib.reload(sys.modules[module_name])
    else:
        imported_module = importlib.import_module(module_name)

    tool_objects = [getattr(imported_module, var_name) for var_name in tool_variable_names]

    return tool_objects

tools_list = gather_tools_from_file()

agent = initialize_agent(
    tools_list, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True)
prompt = "What is 9/3"
response = agent.run(prompt)
print(response)