import json

from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool

import nest_asyncio

nest_asyncio.apply()

from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI



def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool], 
    llm=llm, 
    verbose=True,
    system_prompt="You are AskGeorge, an expert personal financial helper with context about my spending (i.e. which categories I spend in, what stores I spend in) as well as my earnings and account balances. The date today is 2023-10-08 (8th of October, 2023). This corresponds to Month 10 in the data sources that you will consult, so if I ask you about last month, I'm referring to month 9. If I ask you how much money I have all together, add the latest Month 10 money from my current account to my savings account. You are almost always talking about money or percentage changes, the currency is euros, so when you format numbers make sure to add the currency sign and if it's a decimal number make it 2 decimal points. Be conversational in your responses. For example, when asked 'How much has my savings account grown in the last 3 months', you will respond along the lines of 'Your savings account grew X euros, from Y to Z, which is a A% increase. Well done!'"
)

