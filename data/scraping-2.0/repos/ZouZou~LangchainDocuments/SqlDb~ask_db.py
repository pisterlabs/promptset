import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
print('0')
llm = ChatOpenAI()
print('1')
database_user = os.getenv("DATABASE_USERNAME")
database_password = os.getenv("DATABASE_PASSWORD")
database_server = os.getenv("DATABASE_SERVER")
database_db = os.getenv("DATABASE_DB")
print('2')
connection_string = f"mssql+pymssql://{database_user}:{database_password}@{database_server}/{database_db}"
print(connection_string)
db = SQLDatabase.from_uri(connection_string)
print('2.5')
toolkit = SQLDatabaseToolkit(db=db, llm=llm, reduce_k_below_max_tokens=True)
print('3')
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
print('4')
agent_executor.run("Find the top 5 persons")
print('5')