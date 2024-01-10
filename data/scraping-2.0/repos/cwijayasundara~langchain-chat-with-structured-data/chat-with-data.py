import os
import streamlit as st

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

st.title("Chat With Your Data! - Natural Language to SQL ..")
st.text_input("Please Enter Your Query in Plain Text ! ", key="query")
result = agent_executor.run(st.session_state.query)
st.write(result)

# Describe the playlisttrack table
# List the total sales per country. Which country's customers spent the most?
# Who are the top 3 best selling artists?


