import os
from langchain import OpenAI, SerpAPIWrapper
import chainlit as cl
from langchain.agents import Tool, initialize_agent

openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError('OPENAI_API_KEY environment variable is not set.')

# Set SerpApi API key
serp_api_key = os.environ.get('SERPAPI_API_KEY')
if not serp_api_key:
    raise ValueError('SERPAPI_API_KEY environment variable is not set.')

@cl.langchain_factory
def factory():
    search = SerpAPIWrapper()
    tools = [
            Tool(name="Search",
            func=search.run,
            description="Bot answering questions using GPT-3.5, Langchain & Google!",)]
         
    llm=OpenAI(temperature=0, model_name='gpt-3.5-turbo')
    agent = initialize_agent(tools, llm=llm, agent='chat-zero-shot-react-description', verbose=True)
    return agent
