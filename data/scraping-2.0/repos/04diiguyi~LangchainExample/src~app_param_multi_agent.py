# To run: In the current folder: 
# python app_param_multi_agent.py

# This example is a sample that create an agent which helps
# check tire inventory and calculate the sale price of tires. 
# This agent has three tools, one math tool, one checks inventory,
# one fetches tire price. In this demo, price/inventory tools give 
# hard coded responses, however, we can use API tool, 
# SQL tool and even chain tool to 
# customize it for real business logic. 
#
# Please note price tool takes one string as input, and inventory tool
# takes two parameters. That's why we need an agent whose
# type is STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
#
# The backend of llmchain uses OpenAI gpt3.5 chat model which is not a correct approach
# Example response. The answer varys every time, with or without prompt optimization
# > Entering new AgentExecutor chain...
# Thought: I can use the `inventory_api` tool to check if the Issaquah store has 
# enough Goodyear tires in stock. Then, I can use the `Search Price` tool to get the total price of four Goodyear tires.
#
#Action:
#```
#{
#  "action": "inventory_api",
#  "action_input": {
#    "tire": "Goodyear",
#    "store": "Issaquah"
#  }
#}
#```
#
#Observation: There are 10 Goodyear available in store Issaquah.
#Thought:Now that I know there are 10 Goodyear tires available in the Issaquah store, 
#I can use the `Search Price` tool to get the total price of four Goodyear tires.
#
#Action:
#```
#{
#  "action": "Search Price",
#  "action_input": "What is the price of four Goodyear tires?"
#}
#```
#
#Observation: Tire What is the price of four Goodyear tires? is $150 each.
#Thought:Now that I know the price of each tire, I can calculate the total price of four Goodyear tires.
#
#Action:
#```
#{
#  "action": "Final Answer",
#  "action_input": "The total price of four Goodyear tires in the Issaquah store is $600."
#}
#```
#
#> Finished chain.
#The total price of four Goodyear tires in the Issaquah store is $600.
#
#> Entering new AgentExecutor chain...
#Thought: I can use the `inventory_api` tool to check if the Issaquah store 
# has enough Good Year tires in stock. Then, I can use the `Search Price` tool to 
# get the price of one Good Year tire and calculate the total price for 30 tires.
#
#Action:
#```
#{
#  "action": "inventory_api",
#  "action_input": {
#    "tire": "Good Year",
#    "store": "Issaquah"
#  }
#}
#```
#
#Observation: There are 10 Good Year available in store Issaquah.
#Thought:Since there are only 10 Good Year tires available in the Issaquah store, 
#the customer cannot buy 30 tires from that store. I need to inform the customer about this.
#
#Action:
#```
#{
#  "action": "Final Answer",
#  "action_input": "I'm sorry, but there are only 10 Good Year tires available in the Issaquah store. 
# We cannot fulfill your request for 30 tires from that store."
#}
#```
#
#> Finished chain.
#I'm sorry, but there are only 10 Good Year tires available in the Issaquah store. 
# We cannot fulfill your request for 30 tires from that store.

import os

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import StructuredTool
from langchain.chat_models import AzureChatOpenAI

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35
from langchain import LLMMathChain

from tools.tool_price import price_api
from tools.tool_inventory import inventory_api

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

# Check the existence of tools
print(price_api)
print(inventory_api)

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
    StructuredTool.from_function(inventory_api),
]

agent = initialize_agent(tools, chat, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#response = agent.run("I want to buy four good year tires in my local Issaquah store, \
#                     do we have enough in stock and how much is the total price?")

#print(response)

#response = agent.run("I want to buy 30 good year tires in my local Issaquah store, \
#                     do we have enough in stock and how much is the total price?")

#print(response)

# Hallucination error: we do not provide store info, OpenAI hallucinate my_store to fill in the parameter `store`
#Action:
#```
#{
#  "action": "inventory_api",
#  "action_input": {
#    "tire": "goodyear",
#    "store": "my_store"
#  }
#}
#```

#Observation: There are 10 goodyear available in store my_store.
#Thought:Action:
#```
#{
#  "action": "Search Price",
#  "action_input": "goodyear tires"
#}
#```
# ...

response = agent.run("I want to buy four good year tires, \
                     do we have enough in stock and how much is the total price?")

print(response)