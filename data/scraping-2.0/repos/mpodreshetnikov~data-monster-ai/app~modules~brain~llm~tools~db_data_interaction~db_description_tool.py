import re
from typing import Optional
from pydantic import Field

from sqlalchemy.schema import CreateTable

from langchain import SQLDatabase
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    # QueryCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class ListSQLDatabaseWithCommentsTool(ListSQLDatabaseTool):
    cache_key: str = ""
    cache: str = ""

    db_comments_override: list = Field(
        default=None, description="Override for the database comments"
    )

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific table."""
        if self.db._include_tables:
            tables_to_take = self.db._include_tables
        else:
            tables_to_take = self.db._all_tables - self.db._ignore_tables

        # return cached data if possible
        cache_key = str.join(",", tables_to_take)
        if cache_key == self.cache_key:
            return self.cache

        table_strings = []
        for table in tables_to_take:
            # db_comment = self.db._inspector.get_table_comment(table, schema=self.db._schema)["text"] TODO: Временно решили не учитывать комменты из БД
            override_comment = self.__get_override_table_comment(table)
            comment = override_comment  # if override_comment else db_comment
            if comment:
                table_strings.append(f"{table} ({comment})")
            else:
                table_strings.append(table)

        value = ", ".join(table_strings)
        self.cache_key = cache_key
        self.cache = value

        return value

    def __get_override_table_comment(self, table_name: str):
        table_name = table_name.split(".")[-1]
        if not self.db_comments_override:
            return None
        for table in self.db_comments_override:
            if table["name"] == table_name:
                return table["comment"]
        return None

    async def _arun(  # TODO change async
        self,
        tool_input: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self.run(tool_input,run_manager)


class InfoSQLDatabaseWithCommentsTool(InfoSQLDatabaseTool):
    description = """
    Input to this tool is table name, output is the schema: columns, foreign keys and sample rows for the table.
    Be sure that the table actually exists and don't ask more than one table.
    You cannot ask all the tables via this tool. If you want to get all the available tables, use 'list_tables_sql_db' tool.
    """

    db_comments_override: list = Field(
        default=None, description="Override for the database comments"
    )

    def _run(
        self,
        table_name: str,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        table_name = self.__get_table_name(table_name)
        table_names = [table_name]

        try:
            return self.__get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    async def _arun(
        self,
        table_name: str,
        tool_input: str = "",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(table_name, tool_input, run_manager)

    ### Method was taken from InfoSQLDatabaseTool and modified to include column comments ###
    def __get_table_info(self, table_names):
        all_table_names = self.db.get_usable_table_names()
        if table_names is not None:
            if missing_tables := set(table_names) - set(all_table_names):
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self.db._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and (self.db.dialect != "sqlite" or not tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self.db._custom_table_info and table.name in self.db._custom_table_info:
                tables.append(self.db._custom_table_info[table.name])
                continue

            create_table = str(CreateTable(table).compile(self.db._engine))
            table_info = f"{create_table.rstrip()}"

            # add column comments
            for column in table.columns:
                # db_comment = column.comment TODO: Временно решили не учитывать комменты из БД
                override_comment = self.__get_override_column_comment(
                    table.fullname, column.name
                )
                comment = override_comment  # if override_comment else db_comment
                if comment:
                    f_pattern = r"(\s" + column.name + r"\s[^,\n]*)(,?\n?)"
                    s_pattern = r"\1 COMMENT '" + comment + r"'\2"
                    table_info = re.sub(f_pattern, s_pattern, table_info, count=1)
            tables.append(table_info)

            # has_extra_info = (
            #     self.db._indexes_in_table_info or self.db._sample_rows_in_table_info
            # )
            # if has_extra_info:
            #     table_info += "\n\n/*"
            # if self.db._indexes_in_table_info:
            #     table_info += f"\n{self.db._get_table_indexes(table)}\n"
            # if self.db._sample_rows_in_table_info:
            #     table_info += f"\n{self.db._get_sample_rows(table)}\n"
            # if has_extra_info:
            #     table_info += "*/"

        return "\n\n".join(tables)

    def __get_override_column_comment(self, table_name, column_name):
        table_name = table_name.split(".")[-1]
        if not self.db_comments_override:
            return None
        for table in self.db_comments_override:
            if table["name"] == table_name:
                if not table.get("columns"):
                    return None
                for column in table["columns"]:
                    if column["name"] == column_name:
                        return column["comment"]
        return None

    def __get_table_name(self, table_name):
        parts = table_name.split(".")
        if len(parts) == 3:
            db_name, schema, table_name = parts
        elif len(parts) == 2:
            db_name = None
            schema, table_name = parts
        else:
            db_name = None
            schema = None

        # Проверка наличия таблицы
        for table in self.db._metadata.sorted_tables:
            if schema:
                if table.schema == schema and table.name == table_name:
                    return table_name
            elif table.name == table_name:
                return table_name

        return None


class LightweightQuerySQLDataBaseTool(QuerySQLDataBaseTool):
    """Tool for querying SQL databases. Errors are returned simplified."""

    def _run(
        self,
        query: str,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result: str = self.db.run_no_throw(query)
        if "Error:" in result:
            simplified_error_text = result.split("[SQL:")[0].rstrip()
            return simplified_error_text
        return result

    async def _arun(
        self,
        query: str,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result: str = await self.db.arun_no_throw(query)
        if "Error:" in result:
            simplified_error_text = result.split("[SQL:")[0].rstrip()
            return simplified_error_text
        return result


class SQLDatabaseToolkitModified(SQLDatabaseToolkit):
    def get_tools(self, db_comments_override: dict) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            LightweightQuerySQLDataBaseTool(db=self.db),
            InfoSQLDatabaseWithCommentsTool(
                db=self.db, db_comments_override=db_comments_override
            ),
            ListSQLDatabaseWithCommentsTool(
                db=self.db, db_comments_override=db_comments_override
            ),
            # QueryCheckerTool(db=self.db, llm=self.llm),
        ]
