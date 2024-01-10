from typing import List

from decouple import config
from langchain import LLMChain
from langchain.agents import (
    AgentExecutor,
    AgentType,
    StructuredChatAgent,
    ZeroShotAgent,
    initialize_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

OPENAI_API_KEY = config("OPENAI_API_KEY")


def get_structured_chat_agent_from_tools(
    tools: List[BaseTool],
    verbose: bool = False,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
):
    prefix = """Have a conversation with a human, 
    answering the following questions as best you can. 
    You have access to the following tools:
    """
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = StructuredChatAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        llm=ChatOpenAI(
            temperature=temperature, 
            openai_api_key=OPENAI_API_KEY,
             model_name=model_name),  # type: ignore
        prompt=prompt,
    )
    agent = StructuredChatAgent(llm_chain=llm_chain)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
    )
