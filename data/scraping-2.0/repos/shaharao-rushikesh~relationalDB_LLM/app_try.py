# from langchain import SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
# import langchain.utilities.SQLDatabase
# from langchain import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Setting up the api key
import environ
env = environ.Env()
environ.Env.read_env()

API_KEY = env('apikey')


# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:{env('dbpass')}@localhost:5432/Ecom",
)



# setup llm
llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name='gpt-3.5-turbo')

# Create query instruction
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

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