import os
import pyodbc
import tkinter as tk
import tkinter.ttk as ttk
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

# Database connection details for Microsoft SQL Server
DATABASE_SERVER = 'SKUMAR'
DATABASE_NAME = 'Northwind'
# DATABASE_USERNAME = 'your_username'
# DATABASE_PASSWORD = 'your_password'
DRIVER = '{ODBC Driver 18 for SQL Server}'
windowAuthentication_trusted_connection = 'yes'


def setup_openai_api():
    # Set your OpenAI API Key
    os.environ['OPENAI_API_KEY'] = "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"

def setup_database_connection():    

    # Establish a connection to the Microsoft SQL database
    conn_string = f'DRIVER={DRIVER};SERVER={DATABASE_SERVER};DATABASE={DATABASE_NAME};Trusted_Connection={windowAuthentication_trusted_connection};TrustServerCertificate=yes'  
    conn = pyodbc.connect(conn_string)

    return conn

def create_agent_executor():
    
    # Create the agent executor
    # Here, adapt the SQLDatabase.from_uri() method for your Microsoft SQL Server
    # You might need to adjust the URI format based on your pyodbc connection details
    # and how SQLDatabase.from_uri() interprets it.

    # URI for Windows Authentication
    conn_uri = f"mssql+pyodbc://{DATABASE_SERVER}/{DATABASE_NAME}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&Trusted_Connection=yes"
    # Establishing a connection to the Microsoft SQL database using Windows Authentication
    db = SQLDatabase.from_uri(conn_uri)

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
    setup_database_connection()
    agent_executor = create_agent_executor()
    setup_ui(agent_executor)



if __name__ == "__main__":
    main()
