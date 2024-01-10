from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

from tools.base_tool import BaseTool
from utils import logger, prompts
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QueryCheckerTool,
    QuerySQLDataBaseTool,
)

from langchain.tools import Tool

import os


class DatabaseQuery(BaseTool):

    def __init__(self, path, comments, llm) -> None:
        super().__init__(llm)
        self.db = SQLDatabase.from_uri(path)
        self.comments = comments
        logger.debug(f"DatabaseQuery load db from {self.db}")

    def get_tools(self):
        cd = CommentDatabaseTool(self.comments, self.llm)
        return [
            QuerySQLDataBaseTool(db=self.db),
            InfoSQLDatabaseTool(db=self.db),
            ListSQLDatabaseTool(db=self.db),
            QueryCheckerTool(db=self.db, llm=self.llm),
            Tool(name=cd.inference.name,
                 description=cd.inference.desc,
                 func=cd.inference)
        ]


class CommentDatabaseTool(BaseTool):

    def __init__(self, comments, llm) -> None:
        super().__init__(llm)
        self.comments = {}

        for comment in comments:
            with open(comment, "r") as f:
                file_name = ".".join(os.path.basename(comment).split(".")[:-1])
                self.comments[file_name] = f.read()

        logger.debug(f"CommentDatabaseTool load comments from {self.comments}")

    @prompts(
        name="Get comment about table",
        desc="useful when you want to get the comment about a table."
        "The input is a string, which is the table name. "
        "You cannot create a table name. The input must be a valid table name in the database"
        "The output is a string, which is the comment about the table, which will be used to help you understand the table."
    )
    def inference(self, table_name):
        return self.comments.get(table_name, "No comments")
