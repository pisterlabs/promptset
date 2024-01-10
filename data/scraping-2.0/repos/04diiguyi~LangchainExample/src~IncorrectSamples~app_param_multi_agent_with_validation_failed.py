# To run: In the current folder: 
# python app_param_multi_agent_with_validation.py

import os

from langchain.agents import initialize_agent, Tool, AgentType, load_tools
from langchain.tools import StructuredTool, tool
from langchain.llms import AzureOpenAI

from langchain.chat_models import AzureChatOpenAI

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35, Az_Open_Deployment_name_gpt3

from tools.tool_inventory import inventory_api

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

tools.append(StructuredTool.from_function(inventory_api))

# The following tool construction is incorrect since it is not multiple inputs tool
#tools.append(Tool(
#        name="Search Inventory",
#        func=inventory_api,
#        description="Search the inventory information for `tire` in `store`.",
#    ))

llm = AzureOpenAI(
    deployment_name=Az_Open_Deployment_name_gpt3,
    model_name="text-davinci-003", 
)

def inventory_api_validation(input: str) -> str:
    print(f"Input is {input}")

    result = llm(f"We need to check whether the following input has `store` and `tire` information, if not, we need the user for the necessary inputs. Input is {input}")
    return result

tools.append(Tool(
        name = "Inventory Tool Parameter Validation",
        func=inventory_api_validation,
        description="This tool must be called before Tool Search Inventory. The requied parameter `input` is text in message of agent run. If not enough information found, need to ask the user to provide the missing parameters. "
    ))

agent = initialize_agent(tools, chat, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run("I want to buy four good year tires and do we have enough in stock?")
#response = agent.run("I want to buy four good year tires in Issaquah store and do we have enough in stock?")
#response = agent.run("I want to buy 30 good year tires in my local Issaquah store, \
#                     do we have enough in stock?")

print(response)