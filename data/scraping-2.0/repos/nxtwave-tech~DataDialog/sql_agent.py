from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.tools import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
)

from prompts.sql_agent import SQL_AGENT_PREFIX_PROMPT
from langchain.callbacks import StreamlitCallbackHandler

import streamlit as st
import sqlite3
import pandas as pd


DB_FILE_PATH = 'airbnb.db'


class SqlQueryToolkit(SQLDatabaseToolkit):
    def get_tools(self):
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables,"
            " output is the schema and sample rows for those tables."
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            f"replace single quotes with dobule quotes  before calling "
            f"{list_sql_database_tool.name}"
            "Example Input: 'table1, table2, table3'"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct. "
            "return the output in double quotes."
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm,
            description=query_sql_checker_tool_description
        )
        return [
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]


def get_sql_query_from_user_question(user_question):
    st_cb = StreamlitCallbackHandler(st.container())
    agent = _create_sql_agent()

    agent_response = agent({'input': user_question}, callbacks=[st_cb])

    sql_query = _extract_sql_query_from_agent_response(agent_response)

    return sql_query


def get_db_data_with_sql_query(sql_query):
    connection = sqlite3.connect(DB_FILE_PATH)
    query_op_df = pd.read_sql_query(sql_query, connection)
    connection.close()
    return query_op_df


def _extract_sql_query_from_agent_response(agent_response):
    sql_query_str = ''
    for step in agent_response['intermediate_steps']:
        if step[0].tool == 'sql_db_query_checker':
            sql_query_str = step[0].tool_input
            break
    return sql_query_str


def _create_sql_agent():
    llm_model_name = 'gpt-4'
    openai_llm = ChatOpenAI(
        temperature=0, streaming=True, model_name=llm_model_name)

    db_abs_file_path = (Path(__file__).parent / "airbnb.db").absolute()
    db_uri = f"sqlite:////{db_abs_file_path}"
    database = SQLDatabase.from_uri(database_uri=db_uri)
    toolkit = SqlQueryToolkit(db=database, llm=openai_llm)

    agent = create_sql_agent(
        llm=openai_llm,
        toolkit=toolkit,
        verbose=True,
        prefix=SQL_AGENT_PREFIX_PROMPT,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    agent.return_intermediate_steps = True

    return agent
