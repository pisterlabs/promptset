from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from config import my_api_key

db = SQLDatabase.from_uri("sqlite:///chinook.db")
llm = OpenAI(openai_api_key = my_api_key, temperature=0, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

while True:
    q = input('ask me: ')
    db_chain.run(q)
