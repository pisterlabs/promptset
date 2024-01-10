import os

from tools.dbbase import SQLDatabase
from tools.dbchain import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI

with open("openai_api_key.txt", "r") as f:
    api_key = f.read()

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
        temperature=0.4,
    )
        
db = SQLDatabase.from_uri("sqlite:///sayvai.db")

def DatabaseChain(query: str) -> str:
    sql_db_chain = SQLDatabaseChain.from_llm(
    llm = llm,
    db = db,
    )

    return sql_db_chain.run(query)