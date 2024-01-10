import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# tools = load_tools(["serpapi","llm-math"], llm=llm,) 
# agent = initialize_agent(tools, 
#                          llm, 
#                          agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
#                          verbose=True)
# agent.run("What year was Albert Einstein born? What is that year number multiplied by 5?")

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

agent = create_python_agent(tool=PythonREPLTool(),
                         llm=llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True)

python_list = [3,5,2,1,5,7,8,1,9,10]
agent.run(f'''Sort this Python list {python_list}''')