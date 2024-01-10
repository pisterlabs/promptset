import os
import pyodbc
import tkinter as tk
import tkinter.ttk as ttk
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

#from sqlalchemy_sqlany import SAEngine

#import sqlany_dialect  # The custom dialect we defined

# Database connection details for Microsoft SQL Server
DATABASE_SERVER = 'sqldemo'
DATABASE_NAME = 'demo'
DATABASE_USERNAME = 'dba'
DATABASE_PASSWORD = 'sql'
DRIVER_NAME = '{SQL Anywhere 17}'
PORT=2638
DATABASE_HOST = 'localhost'

# Define your DSN name
DSN_NAME = 'sql2023'


def setup_openai_api():
    # Set your OpenAI API Key
    os.environ['OPENAI_API_KEY'] = "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"

def setup_database_connection():    

    # Establish a connection to the Microsoft SQL database
    #conn_string = f'DRIVER={DRIVER};SERVER={DATABASE_SERVER};DATABASE={DATABASE_NAME};UID={DATABASE_USERNAME};PWD={DATABASE_PASSWORD}'  

    conn_string = f'DSN={DSN_NAME}'

    conn = pyodbc.connect(conn_string)

    return conn

def create_agent_executor():
    
    from sqlalchemy import create_engine
    import pyodbc

    # Replace with your DSN name
    dsn_name = 'sql2023'

    # # Build connection string using pyodbc
    # connection_string = pyodbc.connect(f"DSN={dsn_name}")
    # # Create SQLAlchemy engine with pyodbc connection
    
    engine = create_engine(f'sqlalchemy_sqlany://?dsn={dsn_name}')
    db = SQLDatabase(engine,schema='GROUPO')


    #working 
    # conn_uri = f"sqlalchemy_sqlany:///?dsn={dsn_name}"
    # db = SQLDatabase.from_uri(conn_uri)

    

    # Create SQLAlchemy engine
    # engine = SAEngine(dsn="DSN=sql2023")

    # # Connect to SQL Anywhere using SQLDatabase
    # db = SQLDatabase(engine)


    # engine = create_engine('sqlanywhere+pyodbc://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_SERVER}/{DATABASE_NAME}')
    # db = SQLDatabase.from_uri(engine)

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

def setup_ui(agent_executor):
    # Create the UI window
    root = tk.Tk()
    root.title("Chat with your Database")


    # Text entry widget
    entry = ttk.Entry(root, font=("Arial", 14))
    entry.pack(padx=20, pady=20, fill=tk.X)

    # Button callback
    def on_click():
        query = entry.get()
        result = agent_executor.run(query)
        text.delete("1.0", tk.END)
        text.insert(tk.END, result)

    # Button widget
    button = ttk.Button(root, text="Chat", command=on_click)
    button.pack(padx=20, pady=20)

    # Text widget to display the result
    text = tk.Text(root, height=10, width=60, font=("Arial", 14))
    text.pack(padx=20, pady=20)

    # Start the UI event loop
    root.mainloop()

def main():
    setup_openai_api()
    #setup_database_connection()
    agent_executor = create_agent_executor()
    setup_ui(agent_executor)



if __name__ == "__main__":
    main()

