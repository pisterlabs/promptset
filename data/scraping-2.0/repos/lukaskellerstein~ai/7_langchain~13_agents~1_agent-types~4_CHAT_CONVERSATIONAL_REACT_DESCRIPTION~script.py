import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

_ = load_dotenv(find_dotenv())  # read local .env file

# ---------------------------
# First Agent
# ---------------------------
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(temperature=0)


# ZERO_SHOT_REACT_DESCRIPTION
# STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# CONVERSATIONAL_REACT_DESCRIPTION
# CHAT_CONVERSATIONAL_REACT_DESCRIPTION <---------------------
# REACT_DOCSTORE
# SELF_ASK_WITH_SEARCH
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


result = agent_chain.run(input="hi, i am bob")
print(result)

result = agent_chain.run(input="what's my name?")
print(result)

result = agent_chain.run(
    input="what are some good dinners to make this week, if i like thai food?"
)
print(result)

result = agent_chain.run(
    input="tell me the last letter in my name, and also tell me who won the world cup in 1978?"
)
print(result)

result = agent_chain.run(input="whats the current temperature in pomfret?")
print(result)
