import chainlit as cl
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentType, Tool, load_tools
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]
memory = ConversationBufferMemory(memory_key="chat_history")

def chat(query: str):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY,
                 model=MODEL_NAME,
                 temperature=0)
    agent_chain = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory
    )
    return agent_chain.run(input=query)

@cl.on_message # for every user message
async def main(query: str):
    response_text = chat(query)

    # final answer
    await cl.Message(
            content=response_text
        ).send()
    
@cl.on_chat_start
async def start():
    await cl.Message(
            content="Hello there!"
        ).send()