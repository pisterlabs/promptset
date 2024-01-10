# Provided by Samuel Berthollier 
# This is a personal project
#

# Load all dependencies
import os
import google.generativeai as palm

from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

from langchain.agents import create_sql_agent, AgentType, initialize_agent, load_tools, Tool, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

# Set your variables
service_account_file = "your_sa_account_key_file.json"
os.environ['GOOGLE_API_KEY'] = 'your_palm_api_key'
palm.configure(api_key='GOOGLE_API_KEY')

project = "your_project_id"
dataset = "your_dataset"
table = "you_table"

# Define modules
sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'
llm = GooglePalm()
llm.temperature = 0.1
db = SQLDatabase.from_uri(sqlalchemy_url)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the agent
agent_executor = create_sql_agent(
llm=llm,
toolkit=toolkit,
verbose=True,
top_k=1000,
)

# Run the agent in console
agent_executor.run("From what top 3 publisher sources were users coming from?")

# Run the agent in Streamlit
'''
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(prompt, callbacks=[st_callback])
        st.write(response)
'''