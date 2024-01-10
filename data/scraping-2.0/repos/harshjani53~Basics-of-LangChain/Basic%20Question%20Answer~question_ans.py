from langchain.llms import OpenAI
from dotenv import load_dotenv

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()


llModel = OpenAI(temperature=0.6)
tools = load_tools(['wikipedia','llm-math'],llm = llModel)
agent = initialize_agent(
        tools= tools,
        llm = llModel,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )
res = agent.run('What is the difference between bottom speed and top speed of a suzuki Hayabusa?')
print(res)
