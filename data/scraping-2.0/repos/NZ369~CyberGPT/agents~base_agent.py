from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from llms.azure_llms import create_llm
from tools.get_tools import base_tools

llm = create_llm(max_tokens=2000, temp=0.5)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3, return_messages=True)
base_agent = initialize_agent(base_tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)
