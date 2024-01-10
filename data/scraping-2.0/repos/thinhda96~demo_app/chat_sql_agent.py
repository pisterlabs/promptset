from langchain import SQLDatabase
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI

from config import db_config


def create_openai_sqlagent(open_ai_key) -> AgentExecutor:
    # This code initializes a connection to a MySQL database using the provided configuration, creates a ChatOpenAI
    # instance with the GPT-4 model, and then sets up an SQL agent with the database and language model for natural
    # language processing tasks.
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=open_ai_key)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )
    return sql_agent

if __name__ == "__main__":
    sql_agent = create_openai_sqlagent("sk_test_51J")
    val = sql_agent.run("What is the total amount of money spent by user 1?")
    print(val)
