import os
from langchain.tools.base import StructuredTool
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
chat = ChatOpenAI(model_name="gpt-4", temperature=0.2)


from typing import Optional


def get_shopify_insight(shopify_object: Optional[str] = None) -> str:
    """Tool that counts the number of items for a given Shopify data object. Valid shopify_objects include "Customer", "Order", "Product" and "Webhook"."""
    object_name = shopify_object or "Order"
    data = get_data(object_name)
    return str(len(data))


shopify_insights_tool = StructuredTool.from_function(get_shopify_insight)

print(shopify_insights_tool)

# Could call the function directly


# Could simply include it in the tools list of an agent
from langchain.agents import initialize_agent, AgentType

tools = [shopify_insights_tool]
agent_chain = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
