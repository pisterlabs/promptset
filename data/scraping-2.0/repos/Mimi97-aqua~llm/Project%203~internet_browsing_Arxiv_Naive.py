from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI(api_key=api_key)

llm = ChatOpenAI(temperature=0.3)
tools = load_tools(
    ['arxiv']
)

agent_chain = initialize_agent(
    tools,
    llm,
    max_iterations=5,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Algorithm for LLM
    verbose=True,
    handle_parsing_errors=True
)

agent_chain.run(
    'What is LIFO?'
)
