import os
from typing import TypedDict

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from langchain.chains import create_sql_query_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from quart import Blueprint, request

from atypes.error import AppError

page = Blueprint(
    f"{os.path.dirname(__file__).replace(os.path.sep, '-')}-{os.path.splitext(os.path.basename(__file__))[0]}",
    __name__,
)


# Define the expected input type
class Input(TypedDict):
    prompt: str


# Define the expected output type
class Output(TypedDict):
    output: str
    query: str | None
    tokens: int


@page.route("/raw", methods=["POST"])
async def sql_raw() -> Output | tuple[AppError, int]:
    """
    Given a prompt, generate a raw SQL query, execute it as is and return the results.
    The prompt should be a question that can be answered by the database.
    """
    try:
        data: Input = await request.get_json()
        prompt = data["prompt"]

        db = SQLDatabase.from_uri("mysql://root:root@127.0.0.1:3316/retailer_db")
        with get_openai_callback() as cb:
            # Create a chain to generate the SQL query
            chain = create_sql_query_chain(ChatOpenAI(temperature=0), db)
            generated_sql = chain.invoke({"question": prompt}).strip()

            # Run the SQL query as is
            result = db.run_no_throw(generated_sql)

            return {
                "query": generated_sql,
                "output": result,
                "tokens": cb.total_tokens,
            }
    except Exception as e:
        return {"error": str(e)}, 500


@page.route("/agent", methods=["POST"])
async def sql_agent() -> Output | tuple[AppError, int]:
    """
    Given a prompt, make observations by generating, running and then returning the results of a SQL query.
    The prompt should be a question that can be answered by the database.
    """
    try:
        data: Input = await request.get_json()
        prompt = data["prompt"]

        db = SQLDatabase.from_uri("mysql://root:root@127.0.0.1:3316/retailer_db")
        with get_openai_callback() as cb:
            # Create an agent executor
            agent_executor = create_sql_agent(
                llm=OpenAI(temperature=0),
                toolkit=SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0)),
                # verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            result = agent_executor.run(prompt)

            return {
                "query": None,
                "output": result,
                "tokens": cb.total_tokens,
            }
    except Exception as e:
        return {"error": str(e)}, 500
