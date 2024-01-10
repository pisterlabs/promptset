from langchain.tools.base import BaseTool
from sqlalchemy.engine import Engine

from chains.document_tables import run_sql_comment_generator
from chains.metabase import run_graph_creator
from chains.sql_runner import sql_runner


class QuerySQLDataBaseTool(BaseTool):
    """Tool for querying a SQL database."""

    name = "query_sql_db"
    description = """
This tool does not accept SQL queries, just pass always the user input question.
This tool takes the user's input question and retrieves the corresponding result from the database in response to the query.
If the user's input question is not correct, an error message will be returned.
If an error is returned, rewrite the user's input question, check the result, and try again.
    """
    engine: Engine = None

    def _run(self, question: str) -> str:
        """Execute the query, return the results or an error message."""
        result = sql_runner(question.lower(), self.engine, 10)
        return str(result)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")


class SQlCommentGenerator(BaseTool):

    name: str = "sql_comment_generator"
    description: str = """
The result should pass by default. The final answer is the SQL code that generates comments and documentation for a
group of tables. This tool strictly accepts only user input questions and does not support SQL queries. If an incorrect
user input question is provided, an error message will be returned. In such cases, it is necessary to revise the 
user's question, verify the result, and try again. It is expected that all results will pass successfully.
    """
    engine: Engine = None

    def _run(self, question: str) -> str:
        """Execute the query, return the results or an error message."""
        sql_code = run_sql_comment_generator(question.lower(), self.engine)
        return sql_code

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("SQlCommentGenerator does not support async")


class DummyTool(BaseTool):

    name = "None needed"
    description = """
None Input Action
    """

    def _run(self, question: str) -> str:
        return "Pass to the final answer\n"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("DummyTool does not support async")


class MetabaseCreator(BaseTool):

    name: str = "metabase_creator"
    description: str = """
The result should pass by default. The final answer is the url of the dashboard created in metabase.
This tools creates a dashboard in metabase with the user input question, and returns the url of the dashboard.
    """
    engine: Engine = None
    max_queries: int = 10

    def _run(self, question: str) -> str:
        """Execute the query, return the results or an error message."""
        url_dashboard = run_graph_creator(question.lower(), self.engine, self.max_queries)
        return url_dashboard

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("SQlCommentGenerator does not support async")
