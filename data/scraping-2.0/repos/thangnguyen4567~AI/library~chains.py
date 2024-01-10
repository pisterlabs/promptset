from typing import List, TypedDict, Union

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import NoOpOutputParser
from langchain.schema.runnable import Runnable, RunnableParallel
from config.config_vectordb import VectorDB

def _strip(text: str) -> str:
    return text.strip()


class SQLInput(TypedDict):
    """Input for a SQL Chain."""

    question: str


class SQLInputWithTables(TypedDict):
    """Input for a SQL Chain."""

    question: str
    table_names_to_use: List[str]

def get_table_info(question) -> str:
    tables = []
    docs = VectorDB().connect_vectordb('training_ddl').similarity_search(query=question,k=10)
    for value in docs:
        tables.append(value.metadata['table']+'\n')
    final_str = "\n\n".join(tables)
    return final_str

def create_sql_query_chain(
    llm: BaseLanguageModel,
    k: int = 5,
    question: str = ''
) -> Runnable[Union[SQLInput, SQLInputWithTables], str]:
    """Create a chain that generates SQL queries.

    *Security Note*: This chain generates SQL queries for the given database.

        The SQLDatabase class provides a get_table_info method that can be used
        to get column information as well as sample data from the table.

        To mitigate risk of leaking sensitive data, limit permissions
        to read and scope to the tables that are needed.

        Optionally, use the SQLInputWithTables input type to specify which tables
        are allowed to be accessed.

        Control access to who can submit requests to this chain.

        See https://python.langchain.com/docs/security for more information.

    Args:
        llm: The language model to use
        db: The SQLDatabase to generate the query for
        prompt: The prompt to use. If none is provided, will choose one
            based on dialect. Defaults to None.
        k: The number of results per select statement to return. Defaults to 5.

    Returns:
        A chain that takes in a question and generates a SQL query that answers
        that question.
    """
    db_dialect = 'mssql'
    prompt_to_use = SQL_PROMPTS[db_dialect]

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "top_k": lambda _: k,
        "table_info": lambda x: get_table_info(question),
    }
    if "dialect" in prompt_to_use.input_variables:
        inputs["dialect"] = lambda _: (db_dialect, prompt_to_use)
    return (
        RunnableParallel(inputs)
        | prompt_to_use
        | llm.bind(stop=["\nSQLResult:"])
        | NoOpOutputParser()
        | _strip
    )
