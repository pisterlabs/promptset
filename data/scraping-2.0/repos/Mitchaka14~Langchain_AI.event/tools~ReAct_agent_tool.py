from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain import SerpAPIWrapper
import os
import streamlit as st
from my_tools import DataTool, SQLAgentTool, EmailTool

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    os.environ["serpapi_api_key"] = st.secrets["SERPAPI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["serpapi_api_key"] = os.getenv("SERPAPI_API_KEY")
llm = OpenAI(temperature=0)

search = SerpAPIWrapper()
data_tool = DataTool()
sql_agent_tool = SQLAgentTool()
email_sender_tool = EmailTool()
sql_agent_tool.description = ""

tools = [
    data_tool,
    sql_agent_tool,
    email_sender_tool,
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to ask with search..use for realtime questions like time etc and some internet related things.....",
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("tell me about narutos appointment, ,my email is akaelumitchell@gmail.com")
