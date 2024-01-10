from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain import OpenAI

import os
from dotenv import load_dotenv

api_key = os.get_env("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=api_key,
    temperature=0
)

os.environ["WOLFRAM_ALPHA_APPID"] = "your_app_id"

from langchain. utilities.wolfram_alpha import WolframAlphaAPIWrapper

wolfram = WolframAlphaAPIWrapper()
result = wolfram.run("What is 2x+5 = -3x + 7?")
print(result)  # Output: 'x = 2/5'



tools = load_tools(["wolfram-alpha"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

print( agent.run("How many days until the next Solar eclipse") )



tools = load_tools(["wolfram-alpha", "wikipedia"], llm=llm)

agent = initialize_agent(
		tools,
		llm,
		agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
		verbose=True
	)

agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")



