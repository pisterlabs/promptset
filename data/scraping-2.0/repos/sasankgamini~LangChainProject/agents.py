from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
#Building a python interpreter agent
#Import Python REPL tool and instantiate Python agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI

agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True
)




#Execute the Python agent which allows us to have the language model use python code
agent_executor.run("Find the roots (zeros) if the quadratic function 3 * x**2 + 2*x -1")
     