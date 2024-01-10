# To run: In the current folder: 
# python app_param_single_agent.py

# This example is a sample that create an agent which helps calculate the 
# sale price of tires. This agent has two tools, one math tool,
# one fetches tire price. In this demo, price tool gives a hard coded response, 
# however, we can use API tool, SQL tool and even chain tool to 
# customize it for real business logic. 
#
# Please note this agent only support tools with one string input, if you need
# tools with multiple parameters, please refer to app_param_multi_agent.py
#
# The backend of llmchain uses OpenAI gpt3.5 chat model which is not a correct approach
# Example response
# > Entering new AgentExecutor chain...
# I need to find the price of Good Year tires
# Action: Search Price
# Action Input: "Good Year tires price"
# Observation: Tire Good Year tires price is $150 each.
# Thought:I need to calculate the total cost for four tires
# Action: Calculator
# Action Input: 150 x 4
# Observation: Answer: 600
# > Finished chain.
# answer: 600

import os

from langchain import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35
from tools.tool_price import price_api

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

# Check the existence of tools
print(price_api)

## Set up OpenAI as chat LLM
chat = AzureChatOpenAI(deployment_name=Az_Open_Deployment_name_gpt35,
            openai_api_version="2023-03-15-preview", temperature=0)

llm_math_chain = LLMMathChain(llm=chat)

tools = [
    Tool(
        name = "Search Price",
        func=price_api.run,
        description="useful for when you need to answer the price of tires"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
        return_direct=True
    )
]

agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run("How much are four good year tires?")

print(response)
