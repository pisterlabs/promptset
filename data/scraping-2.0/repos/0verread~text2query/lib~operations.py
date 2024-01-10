import os

from langchain.llms import OpenAI
from langchain.chains import SQLDatabaseSequentialChain
from langchain import SQLDatabase

llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])


def get_db_uri(dbtype, user, password, dbname, host=None, port=None):
    # postgres://YourUserName:YourPassword@YourHostname:Port2/YourDatabaseName
    uri = f'{dbtype}://{user}:{password}@{host}:{port}/{dbname}'
    return uri

def run_query(query, db_ins):
    chain = SQLDatabaseSequentialChain(llm, db_ins, verbose=True)
    resp = chain.run(query)
    return resp

def get_query_resp(query, db_uri):
    db_ins = SQLDatabase.from_uri(db_uri)
    resp = run_query(query, db_ins)
    return resp
