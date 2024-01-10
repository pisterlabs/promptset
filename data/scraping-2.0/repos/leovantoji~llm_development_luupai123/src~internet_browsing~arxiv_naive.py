from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0.3)
tools = load_tools(["arxiv"])

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    max_iterations=5,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

agent_chain.run("What's income waterfall model?")
