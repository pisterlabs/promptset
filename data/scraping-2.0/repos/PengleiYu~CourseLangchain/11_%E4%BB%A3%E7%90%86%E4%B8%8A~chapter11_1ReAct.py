from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms.openai import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(tool_names=['serpapi', 'llm-math'], llm=llm)
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("目前市场上玫瑰花的平均价格是多少？如果我在此基础上加价15%卖出，应该如何定价？")
