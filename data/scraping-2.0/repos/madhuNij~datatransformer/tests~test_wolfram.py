from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(['serpapi',"llm-math"],llm=llm)
agent = initialize_agent(tools,llm, agent="zero-shot-description",verbose=True)
agent.run("Who is Taylor Swift's boyfriend?")