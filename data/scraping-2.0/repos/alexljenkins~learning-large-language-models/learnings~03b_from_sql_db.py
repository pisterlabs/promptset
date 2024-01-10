"""
This is an example of allowing the LLM to query your database/data directly.
Easily swap out to different data sources or llms - but be careful as
the LLM can run code directly against the data (ie drop table).
"""
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

load_dotenv()

dburi = "sqlite:///datasets/rick_data/full_script.db"
db = SQLDatabase.from_uri(dburi)
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

db_chain.run("Describe the full_script table for me, what columns and content does it contain?")

db_chain.run("Rank the characters from most to least lines of dialogue, including only those with more than 50 lines, ignore the character 'DIRECTION'. Do not set any other limits.")
db_chain.run("Who is Morty in love with?")
