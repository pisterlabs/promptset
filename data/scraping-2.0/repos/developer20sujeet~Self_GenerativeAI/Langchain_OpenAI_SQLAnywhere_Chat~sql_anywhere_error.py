import os
import pyodbc
import tkinter as tk
import tkinter.ttk as ttk
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from sqlalchemy import create_engine

# Database connection details for Microsoft SQL Server
DATABASE_SERVER = 'sqldemo'
DATABASE_NAME = 'demo'
DATABASE_USERNAME = 'dba'
DATABASE_PASSWORD = 'sql'
DRIVER = '{SQL Anywhere 17}'
PORT=2638 
DATABASE_HOST = 'localhost'


def create_agent_executor():
   
   # engine = create_engine(f'sqlanywhere+pyodbc://{DATABASE_USERNAME}:{DATABASE_USERNAME}@{DATABASE_SERVER}:{PORT}/{DATABASE_NAME}?driver=sql+anywhere+17')
    
    # Create the SQLAlchemy engine using the SQL Anywhere dialect
    
    engine = create_engine(f'sqlalchemy_sqlany://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}')
    db = SQLDatabase(engine) #error line 
    
    # conn_uri = f"sqlalchemy_sqlany://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}"      
    # db = SQLDatabase.from_uri(conn_uri) #error line 
    




    # Instantiate your language model here
    llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'], temperature=0)

    # Create the toolkit with the language model and database
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create the SQL agent
    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True
    )

    return agent_executor


def main():
   
    agent_executor = create_agent_executor()
   

if __name__ == "__main__":
    main()
