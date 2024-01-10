import chainlit as cl
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI
from langchain.agents import AgentType, Tool, load_tools, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import SerpAPIWrapper
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

memory = ConversationBufferMemory(memory_key="chat_history")

def chat(query: str):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY,
                 model=MODEL_NAME,
                 max_tokens=2048,
                 temperature=0)
    toolkit = load_tools(
        ["serpapi", "open-meteo-api", "news-api", 
         "python_repl", "wolfram-alpha"], 
         llm=llm, 
         serpapi_api_key=os.getenv('SERPAPI_API_KEY'),
         news_api_key=os.getenv('NEWS_API_KEY'),
         tmdb_bearer_token=os.getenv('TMDB_BEARER_TOKEN')
         )
    ## toolkit += [DuckDuckGoSearchRun()]
    agent_chain = initialize_agent(
        tools=toolkit, 
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