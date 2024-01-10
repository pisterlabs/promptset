"""Query DB using ChatGPT.

https://python.langchain.com/docs/use_cases/qa_structured/sql
https://python.langchain.com/docs/expression_language/cookbook/sql_db
https://python.langchain.com/docs/use_cases/question_answering/
https://python.langchain.com/docs/get_started/quickstart
https://dev.to/ngonidzashe/speak-your-queries-how-langchain-lets-you-chat-with-your-database-p62
"""
from dotenv import load_dotenv
import os

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
import pandas as pd

from lib.db.sql.helper import db_uri, load_query_as_df

current_file_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(current_file_directory, "../../../.env")) # noqa
load_dotenv(dotenv_path=env_path)
openai_api_key = os.getenv("OPENAI_API_KEY")

# https://python.langchain.com/docs/integrations/toolkits/sql_database#initialization
# https://python.langchain.com/docs/use_cases/qa_structured/sql#case-3-sql-agents
llm = ChatOpenAI(
    openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0
)
db = SQLDatabase.from_uri(db_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm) # note: example uses OpenAI not ChatOpenAI, unsure if this matters?
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


def query_chatgpt(query: str) -> None:
    agent_executor.run(query)


if __name__ == "__main__":
    query = input("Please enter a question:\t")
    query_chatgpt(query)
