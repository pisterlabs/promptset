import os
#from dotenv import load_dotenv
import openai
import langchain

os.environ["OPENAI_API_KEY"] =""
os.environ["SQL_SERVER_USERNAME"] = "" 
os.environ["SQL_SERVER_ENDPOINT"] = ""
os.environ["SQL_SERVER_PASSWORD"] = ""  
os.environ["SQL_SERVER_DATABASE"] = ""

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from langchain.sql_database import SQLDatabase

db_config = {  
    'drivername': 'mssql+pyodbc',  
    'username': os.environ["SQL_SERVER_USERNAME"] + '@' + os.environ["SQL_SERVER_ENDPOINT"],  
    'password': os.environ["SQL_SERVER_PASSWORD"],  
    'host': os.environ["SQL_SERVER_ENDPOINT"],  
    'port': 1433,  
    'database': os.environ["SQL_SERVER_DATABASE"],  
    'query': {'driver': 'ODBC Driver 17 for SQL Server'}  
}  

db_url = URL.create(**db_config)
db = SQLDatabase.from_uri(db_url)

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent
#from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

# Page title
st.set_page_config(page_title='🦜🔗 Ask the SQLSaturday App')
st.title('📎Ask the SQLSaturda Oslo DB with Clippy!')


def generate_response(input_query):
    llm = OpenAI(temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    response = agent_executor.run(input_query)
    return st.success(response)

question_list = [
  'How many rows are there?',
  'What kind of tables are here?',
  'How many are called John?',
  'Other']
query_text = st.selectbox('Select an example query:', question_list)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (query_text))

# App logic
if query_text == 'Other':
  query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...')
if not openai_api_key.startswith('sk-'):
  st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-'):
  st.header('Output')
  generate_response(query_text)





