#Conversation Agent
'''
This notebook walks through using an agent optimized for conversation. Other agents are often optimized for using 
tools to figure out the best response, which is not ideal in a conversational setting where you may want the agent 
to be able to chat with the user as well.
This is accomplished with a specific type of agent (conversational-react-description) which expects to be used with a memory component.
'''

from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
import os
os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"
serpapi_key="serpapi_key"

def conversation_agent():
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    tools = [
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm=OpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

    agent_chain.run(input="hi, i am bob")

    agent_chain.run(input="what's my name?")

    agent_chain.run("what are some good dinners to make this week, if i like thai food?")

    agent_chain.run(input="tell me the last letter in my name, and also tell me who won the world cup in 1978?")

    agent_chain.run(input="whats the current temperature in pomfret?")
conversation_agent()
