import sys
from langchain import OpenAI
from langchain import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain

import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env('OPENAI_API_KEY')

if API_KEY == "":
	print("Missing OpenAPI key")
	exit()

if len(sys.argv) < 2:
	print("Missing db connection string.  Example 'postgresql+psycopg2://postgres:1234@localhost:6667/mydb'")
	exit()

dbstring = sys.argv[1]

print("Using OpenAPI with key ["+API_KEY+"] and Database ["+dbstring+"]")

# Setup database
db = SQLDatabase.from_uri(
    dbstring,
)

# setup llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
	temperature=0,
	# max_tokens=1000,
	openai_api_key=API_KEY)

# Create db chain
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""

# Setup the database chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                question = QUERY.format(question=prompt)
                print(db_chain.run(question))
            except Exception as e:
                print(e)

get_prompt()
