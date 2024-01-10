name = "💬 Conversational Agent"
arguments = ["vectorstore","chat_memory"]
annotated = ["OpenAI Agent","Chat LLM","Retriever Tool"]

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from models.llms.llms import *
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

def agent(vectorstore,chat_memory):

    tool = create_retriever_tool(
        vectorstore.as_retriever(), 
        "search",
        "Searches and returns documents based on knowledge base."
    )
    tools = [tool]

    memory_key = "chat_history"
    system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Only use the tools to search for "
                "relevant information. Answers must be based ONLY on the tools"
            )
    )
    prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
        )
    agent = OpenAIFunctionsAgent(llm=chat_llm, tools=tools, prompt=prompt)
    
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=chat_llm,chat_memory=ChatMessageHistory(messages=chat_memory))
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory,
                                    return_intermediate_steps=True)
    
    return agent_executor





