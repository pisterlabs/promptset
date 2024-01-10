"""
wandb documentation to configure wandb using env variables
https://docs.wandb.ai/guides/track/advanced/environment-variables

runs on local server;
wandb server start
"""


import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

# set wandb project name
os.environ["WANDB_PROJECT"] = "langchain-tracing"


from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.callbacks import wandb_tracing_enabled

llm = OpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
agent = initialize_agent(
  tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

questions = [
    "Find the square root of 5.4.",
    "What is 3 divided by 7.34 raised to the power of pi?",
    "What is the sin of 0.47 radians, divided by the cube root of 27?",
    "what is 1 divided by zero"
]

with wandb_tracing_enabled():
    for question in questions:
      try:
        answer = agent.run(question)
        print(answer)
      except Exception as e:
        print(e)
        pass


# remove loop to make a single prompt wanbd trace