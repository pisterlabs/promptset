from dotenv import load_dotenv
load_dotenv()

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.callbacks import LLMonitorCallbackHandler

handler = LLMonitorCallbackHandler()

llm = OpenAI()
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

agent.run(
"What is the approximate result of 78 to the power of 5?",

  callbacks=[handler], # Add the handler to the agent
  metadata={ "agentName": "SuperCalculator" }, # Identify the agent in the LLMonitor dashboard
  )