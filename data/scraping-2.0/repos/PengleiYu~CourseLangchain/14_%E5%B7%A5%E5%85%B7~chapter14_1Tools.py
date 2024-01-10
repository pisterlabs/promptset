from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()
tools = load_tools(['arxiv'])

agent_chain = initialize_agent(tools=tools,
                               llm=llm,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=True)
result = agent_chain.run("介绍一下2005.14165这篇论文的创新点?")
print(result)
