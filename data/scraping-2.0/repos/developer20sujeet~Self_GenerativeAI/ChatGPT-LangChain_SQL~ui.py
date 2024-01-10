
import os

os.environ['OPENAI_API_KEY'] = ""
import sqlite3
import tkinter as tk
import tkinter.ttk as ttk
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor


#==================================================================================================================
#==================================================================================================================
#==================================================================================================================

# Connect to a new database file
conn = sqlite3.connect('Chinook.db')

# Open the SQL script file and read its contents
with open('./Chinook_Sqlite.sql', 'r', encoding='utf8') as f:
    sql_script = f.read()

# Execute the SQL script in the new database
conn.executescript(sql_script)

# Close the database connection
conn.close()



#==================================================================================================================
#==================================================================================================================
#==================================================================================================================

# Create the agent executor

#improve the LLM’s ability to create optimal queries by example rows in a SELECT statement following the CREATE TABLE description resulted in consistent performance improvements.  they found that providing 3 rows was optimal
    # db = SQLDatabase.from_uri(
    # 	"sqlite:///./Chinook.db",
    # 	include_tables=['Track'], # including only one table for illustration
    # 	sample_rows_in_table_info=3
    # )
    # print(db.table_info)

db = SQLDatabase.from_uri("sqlite:///./Chinook.db")

# Make sure to instantiate your language model here
llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'], temperature=0)

# Pass the language model (llm) when creating the toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)


#==================================================================================================================
#==================================================================================================================
#==================================================================================================================


# Create the UI window
root = tk.Tk()
root.title("Chat with your Tabular Data")

# Create the text entry widget
entry = ttk.Entry(root, font=("Arial", 14))
entry.pack(padx=20, pady=20, fill=tk.X)

# Create the button callback
def on_click():
    # Get the query text from the entry widget
    query = entry.get()

    # Run the query using the agent executor
    result = agent_executor.run(query)

    # Display the result in the text widget
    text.delete("1.0", tk.END)
    text.insert(tk.END, result)

# Create the button widget
button = ttk.Button(root, text="Chat", command=on_click)
button.pack(padx=20, pady=20)

# Create the text widget to display the result
text = tk.Text(root, height=10, width=60, font=("Arial", 14))
text.pack(padx=20, pady=20)

# Start the UI event loop
root.mainloop()
