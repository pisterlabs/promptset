from typing import Optional, List

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

import os

from langchain.tools import BaseSQLDatabaseTool, BaseTool, QuerySQLDataBaseTool, InfoSQLDatabaseTool, \
    ListSQLDatabaseTool, QuerySQLCheckerTool

os.environ['OPENAI_API_KEY'] = "sk-UjRJKjGVOKzud8xxxxxxxxtOoJcw655VCT"
os.environ["OPENAI_PROXY"] = "http://127.0.0.1:7890"


class QuerySQLTableTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL Table."""

    name = "sql_table_query"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the table.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)

    async def _arun(
            self,
            query: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")

class SQLDatabaseToolkit2(SQLDatabaseToolkit):

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            "'xxxx' in 'field list', using schema_sql_db to query the correct table "
            "fields."
        )
        query_sql_table_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the table, like row count. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            "'xxxx' in 'field list', using schema_sql_db to query the correct table "
            "fields."
        )
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling list_tables_sql_db "
            "first! Example Input: 'table1, table2, table3'"
        )
        return [
            QuerySQLDataBaseTool(
                db=self.db, description=query_sql_database_tool_description
            ),
            QuerySQLTableTool(
                db=self.db, description=query_sql_table_tool_description
            ),
            InfoSQLDatabaseTool(
                db=self.db, description=info_sql_database_tool_description
            ),
            ListSQLDatabaseTool(db=self.db),
            QuerySQLCheckerTool(db=self.db, llm=self.llm),
        ]


db = SQLDatabase.from_uri("sqlite:////home/tianjiqx/qx/sqllite/Chinook.db")
toolkit = SQLDatabaseToolkit2(db=db, llm=OpenAI(temperature=0))

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""


SQL_SUFFIX = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query. Answer questions based on the most relevant tables.
Generally, each table should obtain information independently
{agent_scratchpad}"""


frist_promt= """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.


sql_db_query: Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', using schema_sql_db to query the correct table fields.
sql_table_query: Input to this tool is a detailed and correct SQL query, output is a result from the table, like row count. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', using schema_sql_db to query the correct table fields.
sql_db_schema: Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling list_tables_sql_db first! Example Input: 'table1, table2, table3'
sql_db_list_tables: Input is an empty string, output is a comma separated list of tables in the database.
sql_db_query_checker: 
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [sql_db_query, sql_table_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query. Answer questions based on the most relevant tables.
Generally, each table should obtain information independently
{agent_scratchpad}
"""


agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0.7),
    toolkit=toolkit,
    suffix=SQL_SUFFIX,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


# agent_executor.run("Describe the playlisttrack table")

agent_executor.run("总共有多少张表,数据量最多的表是那张，有多少行, PlaylistTrack 表有多少行")
