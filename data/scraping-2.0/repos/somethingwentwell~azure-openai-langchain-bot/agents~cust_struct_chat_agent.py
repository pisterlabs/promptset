from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser, StructuredChatAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re

PREFIX = """
Respond to the human as helpfully and accurately as possible. 

You have access to the following tools:

"""

# PREFIX = """

# You are GitLab Assistant.

# You CAN ONLY answer questions which is related to GitLab.

# If there's any questions are not related to GitLab or small talk, you must reply "I don't know".

# Respond to the human as helpfully and accurately as possible. 

# You have access to the following tools:

# """

SUFFIX = """

Previous conversation history:
{chat_history}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:
"""


class CustStructChatAgent():
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
        super().__init__()

    def setup(self):
        prompt = StructuredChatAgent.create_prompt(
            tools=self.tools,
            prefix=PREFIX,
            suffix=SUFFIX,
            input_variables=["input", "agent_scratchpad", "chat_history"])
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.tools]
        agent = StructuredChatAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        return agent