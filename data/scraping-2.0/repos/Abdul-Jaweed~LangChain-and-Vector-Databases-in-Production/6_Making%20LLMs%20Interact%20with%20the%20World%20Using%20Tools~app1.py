import os

os.environ["OPENAI_API_KEY"] = "<YOUR-OPENAI-API-KEY>"
os.environ["GOOGLE_API_KEY"] = "<YOUR-GOOGLE-API-KEY>"
os.environ["GOOGLE_CSE_ID"] = "<YOUR-GOOGLE-CSE-ID>"


# As a Standalone utility:

from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

search.results("What is the capital of Spain ?", 3)



from langchain.agents import initialize_agent, load_tools, AgentType

tools = load_tools(["google-search"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent("What is the national drink in Spain ?"))