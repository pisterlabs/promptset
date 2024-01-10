from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI

from src.tools import (
    search_tool,
    wikipedia_tool,
    wolfram_tool,
    calculator_tool,
    newsapi_tool,
)

LLM = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.0)
MEMORY_BUFFER_WINDOW = 10


def build_agent(messages: list):
    memory = _build_memory(messages)
    agent = initialize_agent(
        [search_tool, wikipedia_tool, wolfram_tool, calculator_tool, newsapi_tool],
        LLM,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        memory=memory,
    )
    return agent


def _build_memory(messages: list):
    memory = ConversationBufferWindowMemory(
        k=MEMORY_BUFFER_WINDOW, memory_key="chat_history", return_messages=True
    )
    for message in messages[-MEMORY_BUFFER_WINDOW:]:
        if isinstance(message, AIMessage):
            memory.chat_memory.add_ai_message(message.content)
        elif isinstance(message, HumanMessage):
            memory.chat_memory.add_user_message(message.content)
    return memory
