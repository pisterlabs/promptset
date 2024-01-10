from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
import os

OPENAI_API_KEY = ""
SERPAPI_API_KEY = ""
SOURCE_DIRECTORY = 'C:\code\langchain\libs\langchain\_sources'

print('starting')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = SERPAPI_API_KEY

def calculate(i):
    return 20

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="useful for solving math problems",
    )    
]

llm = OpenAI(temperature=0)

agent_executor = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent_executor.invoke(
    {
        "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    }
)
