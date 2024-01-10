from langchain import OpenAI
from langchain.chains.postgre_sql.base import PostgreSQLChain, PostgreSQLSequentialChain
from langchain.postgres import PostgreSQL


db: PostgreSQL = PostgreSQL(user="postgres", password="guNagaNa1", host="127.0.0.1", port="5432", database="chinook", schema="public")

llm: OpenAI = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

db_chain: PostgreSQLChain = PostgreSQLChain(llm=llm, database=db, verbose=True)

db_chain.run("How many employees are there?")
