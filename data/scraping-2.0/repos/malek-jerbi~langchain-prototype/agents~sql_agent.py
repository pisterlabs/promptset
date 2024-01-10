from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def sql_agent(prompt):
    db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))

    toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))

    agent = create_sql_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    response = agent.run("student_id: 3 " + prompt)

    return response
