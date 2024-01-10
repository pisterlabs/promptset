# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/11/27 14:27

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool,AgentType
from langchain.utilities import WikipediaAPIWrapper


model_args = dict(
    openai_api_key="112233",
    openai_api_base="http://192.168.2.180:8081/v1",
    model_name="qwen-chat-7b-int4",
)

llm = ChatOpenAI(temperature=0, **model_args)

wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you need to get information from wikipedia about a single topic"
    ),
]

agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

output = agent_executor.run("Can you please provide a quick summary of Napoleon Bonaparte? \
                          Then do a separate search and tell me what the commonalities are with Serena Williams")


print (output)