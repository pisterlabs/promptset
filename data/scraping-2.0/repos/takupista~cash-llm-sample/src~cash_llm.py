import os
from dotenv import dotenv_values

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

# Get a dictionary of .env variables
# ref: https://ioflood.com/blog/python-dotenv-guide-how-to-use-environment-variables-in-python/
# ref: https://pypi.org/project/python-dotenv/
config = dotenv_values()
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# connect to test database
# ref: https://python.langchain.com/docs/use_cases/qa_structured/sql
db = SQLDatabase.from_uri(f"sqlite:////{config['DB_PATH']}")

# Create a SQL agent using ‘gpt-4’ model with ZERO_SHOT_REACT_DESCRIPTION
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0, model="gpt-4"))

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.run("11月中の利用履歴をもとに支出合計を説明してから、利用先ごとの割合（支出合計に対する割合）で支出傾向について補足してください。")
