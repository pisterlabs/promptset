#C:\Users\XXX\AppData\Local\Programs\Python\Python310\Lib\site-packages\sqlalchemy_sqlany\base.py
import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.engine.url import URL

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase
from pydantic import root_validator
import sqlalchemy_pymssql_sybase



dsn_name='sql2023'
os.environ['OPENAI_API_KEY'] = "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"

schema_name='GROUPO'
#engine = create_engine("sybase+pyodbc://dba:sql@sql2023") #sqlalchemy-sybase
engine = create_engine(f'sqlalchemy_sqlany://?dsn={dsn_name}')
 
#engine  = create_engine("sybase+pymssql://dba:sql@sql2024/demo")


metadata_obj = MetaData(schema=schema_name)

db = SQLDatabase(engine,schema_name)

#db.run('select * from customers')


llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'], temperature=0)

    # Create the toolkit with the language model and database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)


    # Create the SQL agent
agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True
    )

print("table list--  ")
agent_executor.run("give me list of table in db")

#========================================

# engine_url = URL.create(
#     "sybase+pyodbc",
#     username="dba",
#     password="sql",
#     host="sql2024",
#     database="GROUPO"
# )
# engine = create_engine(engine_url)



# @event.listens_for(engine, "connect", insert=True)
# def set_current_schema(dbapi_connection, connection_record):
#     cursor_obj = dbapi_connection.cursor()
#     cursor_obj.execute("SET SCHEMA %s" % schema_name)
#     #cursor_obj.execute("ALTER SESSION SET CURRENT_SCHEMA=%s" % schema_name)
#     cursor_obj.close()
