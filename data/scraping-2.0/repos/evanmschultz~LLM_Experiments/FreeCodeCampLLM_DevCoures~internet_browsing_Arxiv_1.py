from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType, AgentExecutor
import os


# If the parser is erroring out, remember to set temperature to a higher value!!!

llm = ChatOpenAI(temperature=0.3)
tools: list = load_tools(["arxiv"])

agent_chain: AgentExecutor = initialize_agent(
    tools=tools,
    llm=llm,
    max_iterations=5,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

agent_chain.run(
    "What does the 'Attention is All You Need' paper introduce?",
)
