from langchain import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain

from util import conf

pg_uri = f"postgresql+psycopg2://{conf.postgresql['username']}:{conf.postgresql['password']}@{conf.postgresql['host']}:{conf.postgresql['port']}/{conf.postgresql['database']}"
db = SQLDatabase.from_uri(pg_uri)
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True, top_k=3)

PROMPT = """ 
Given an input question, first create a syntactically correct postgresql query to run,  
then look at the results of the query and return the answer.
The question: {question}
"""

question = "年纪最大的几个人?"
db_chain.run(query=PROMPT.format(question=question))
