from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import os
os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"



def sql_agent():
    db = SQLDatabase.from_uri("sqlite:///C:/Program Files/SQLiteStudio/mydb.db")
    llm = OpenAI(temperature=0)
    toolkit = SQLDatabaseToolkit(db=db,llm=llm)

    agent_executor = create_sql_agent(
        llm=OpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True
    )

    #Example: describing a table
    agent_executor.run("Describe the mydb table")

sql_agent()