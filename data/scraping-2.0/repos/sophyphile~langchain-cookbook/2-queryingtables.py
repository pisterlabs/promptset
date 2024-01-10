# Querying Tabular Data
# Super powerful to query this data with LangChain and pass it through an LLM
# Use cases: Use LLMs to query data about users, do data analysis, get real-time information from your DBs.
# Further reading: Agents + Tabular Data

# We will query an SQLite DB with natural language - specifically the San Francisco Trees dataset

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Specify where data is and get connection ready
sqlite_db_path = 'data/San_Francisco_Trees.db'
db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

# Create a chain that takes our LLM and DB. 
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain.run("How many species of trees are there in San Francisco?")

# Confirm Answer with Pandas
import sqlite3
import pandas as pd

# Connect to the SQLite database
connection = sqlite3.connect(sqlite_db_path)

# Define your SQL query
query = "SELECT count(distinct qSpecies) FROM SFTrees"

# Read the SQL query into a Pandas DataFrame
df = pd.read_sql_query(query, connection)

# Close the connection
connection.close()

# Display the result in the first column first cell
print(df.iloc[0,0])




