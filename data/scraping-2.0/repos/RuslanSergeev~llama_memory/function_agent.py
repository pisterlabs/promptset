import os

from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

def bash_shell(command: str) -> str:
    """Run a bash command and return its output as a string"""
    return os.popen(command).read()

def python_fun(script: str) -> None:
    """
    Run a python script, save outut in `result` variable
    args:
        script: a string of python code
    return:
        local variable named `result`
    """
    # execute the argument as a python script
    print ("executing python script: ---")
    print (script)
    print ("---")
    loc = {}
    exec(script, {}, loc)
    return loc.get("result", None)

bash_tool = FunctionTool.from_defaults(fn=bash_shell)
python_tool = FunctionTool.from_defaults(fn=python_fun)

llm = OpenAI(model="gpt-4", temperature=0.1)
agent = OpenAIAgent.from_tools(
    [bash_tool, python_tool],
    llm=llm, 
    verbose=True
)

response = agent.chat("Whats the current rubble/euro course? Use tools to get current values. Show how do you get the answer. Use python tool and requests lib to get the answer.")
print(str(response))
