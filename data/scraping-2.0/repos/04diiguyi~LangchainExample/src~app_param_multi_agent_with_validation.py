# To run: In the current folder: 
# python app_param_multi_agent_with_validation.py

import os

from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.tools import StructuredTool, tool
from langchain.llms import AzureOpenAI

from langchain.chat_models import AzureChatOpenAI

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35, Az_Open_Deployment_name_gpt3

from tools.tool_inventory_with_validation import inventory_api_v2

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

## Set up OpenAI as chat LLM
chat = AzureChatOpenAI(deployment_name=Az_Open_Deployment_name_gpt35,
            openai_api_version="2023-03-15-preview", temperature=0)

tools = load_tools(
    ["human"]
)

tools.append(StructuredTool.from_function(inventory_api_v2))

agent = initialize_agent(tools, chat, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#> Entering new AgentExecutor chain...
#Action:
#```
#{
#  "action": "inventory_api_v2",
#  "action_input": "Do we have enough Goodyear tires in stock?"
#}
#```
#prompt: Extract `tire` and `store` information from Do we have enough Goodyear tires in stock? and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}.
#inventory_api_extract_info:
#
#{"store": "", "tire": "Goodyear"}
#
#Observation: Please ask human to provide `store` information
#Thought:What is the `store` information?
#Action:
#```
#{
#  "action": "Human",
#  "action_input": "Can you please provide me with the `store` information?"
#}
#```
#
#Can you please provide me with the `store` information?
#Redmond
#
#Observation: Redmond
#Thought:Action:
#```
#{
#  "action": "inventory_api_v2",
#  "action_input": "Do we have enough Goodyear tires in stock at Redmond?"
#}
#```
#
#prompt: Extract `tire` and `store` information from Do we have enough Goodyear tires in stock at Redmond? and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}.
#inventory_api_extract_info:
#
#{"store": "Redmond", "tire": "Goodyear"}
#
#Observation: There are 10 Goodyear available in store Redmond.
#Thought:Based on the inventory information, there are 10 Goodyear tires available in the Redmond store.
#
#> Finished chain.
#Based on the inventory information, there are 10 Goodyear tires available in the Redmond store.

###response = agent.run("I want to buy four goodyear tires and do we have enough in stock?")
##################################################################################################

#> Entering new AgentExecutor chain...
#Action:
#```
#{
#  "action": "inventory_api_v2",
#  "action_input": "How many Goodyear tires are available in the Issaquah store?"
#}
#```
#prompt: Extract `tire` and `store` information from How many Goodyear tires are available in the Issaquah store? and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}.
#inventory_api_extract_info:
#
#{"store": "Issaquah", "tire": "Goodyear"}
#
#Observation: There are 10 Goodyear available in store Issaquah.
#Thought:The human wants to buy four Goodyear tires from the Issaquah store and wants to know if there are enough tires in stock. I can use the `inventory_api_v2` tool to check the inventory information for Goodyear tires in the Issaquah store.
#
#Action:
#```
#{
#  "action": "inventory_api_v2",
#  "action_input": "How many Goodyear tires are available in the Issaquah store?"
#}
#```
#
#prompt: Extract `tire` and `store` information from How many Goodyear tires are available in the Issaquah store? and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}.
#inventory_api_extract_info:
#
#{"store": "Issaquah", "tire": "Goodyear"}
#
#Observation: There are 10 Goodyear available in store Issaquah.
#Thought:Based on the previous observation, there are 10 Goodyear tires available in the Issaquah store. The human wants to buy four tires, so there should be enough in stock.
#
#Action:
#```
#{
#  "action": "Final Answer",
#  "action_input": "Yes, there are enough Goodyear tires in stock at the Issaquah store to buy four."
#}
#```
#
#> Finished chain.
#Yes, there are enough Goodyear tires in stock at the Issaquah store to buy four.
response = agent.run("I want to buy four goodyear tires in Issaquah store and do we have enough in stock?")
##################################################################################################

#> Entering new AgentExecutor chain...
#Action:
#```
#{
#  "action": "Human",
#  "action_input": "Can you please provide me with the store name and tire brand you are interested in?"
#}
#```
#
#Can you please provide me with the store name and tire brand you are interested in?
#response = agent.run("I want to buy four tires and do we have enough in stock?")
##################################################################################################

#> Entering new AgentExecutor chain...
#Action:
#```
#{
#  "action": "inventory_api_v2",
#  "action_input": "Do we have enough tires in stock in Redmond store?"
#}
#```
#prompt: Extract `tire` and `store` information from Do we have enough tires in stock in Redmond store? and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}.
#inventory_api_extract_info:
#
#{"store": "Redmond", "tire": ""}
#
#Observation: Please ask human to provide `tire` information
#Thought:What type of tire are you looking for? This information is required to check the inventory.
#
#Action:
#```
#{
#  "action": "Human",
#  "action_input": "What type of tire are you looking for?"
#}
#```
#
#What type of tire are you looking for?
#goodyear
#
#Observation: goodyear
#Thought:Thank you for providing the tire information.
#
#Action:
#```
#{
#  "action": "inventory_api_v2",
#  "action_input": "Do we have enough Goodyear tires in stock in Redmond store?"
#}
#```
#
#prompt: Extract `tire` and `store` information from Do we have enough Goodyear tires in stock in Redmond store? and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}.
#inventory_api_extract_info:
#
#{"store": "Redmond", "tire": "Goodyear"}
#
#Observation: There are 10 Goodyear available in store Redmond.
#Thought:We have 10 Goodyear tires available in the Redmond store.
#
#Action:
#```
#{
#  "action": "Final Answer",
#  "action_input": "Yes, we have enough Goodyear tires in stock. There are 10 available in the Redmond store."
#}
#```
#
#> Finished chain.
#Yes, we have enough Goodyear tires in stock. There are 10 available in the Redmond store.
###response = agent.run("I want to buy four tires and do we have enough in stock? I am in Redmond store.")

print(response)