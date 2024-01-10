__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from src.llm_langchain.llmbaseagent import LLMBaseAgent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from typing import AnyStr, TypeVar, Any, List, Dict

Instancetype = TypeVar('Instancetype', bound='LLMToolkitAgent')


class LLMToolkitAgent(LLMBaseAgent):
    pandas_df_agent_name = "pandas_dataframe"
    sql_agent_name = "sql_database"
    csv_agent_name = "csv"
    spark_sql_agent_name = "spark_sql"

    """
        Wrapper for agent which rely on LangChain tool kits suc as Pandas dataframe
            file system, SQL database, Spark query, Json, CSV file loading..

        :param chat_handle Handle or reference to the large language model
        :param agent LangChain agent executor
    """
    def __init__(self, chat_handle: ChatOpenAI, agent: AgentExecutor, cache_model: AnyStr):
        super(LLMToolkitAgent, self).__init__(chat_handle, agent, cache_model)

    @classmethod
    def build_from_toolkit(cls,
                           chat_handle: ChatOpenAI,
                           agent_name: AnyStr,
                           argument: Any) -> Instancetype:
        """
            Constructor to instantiate the agent from toolkits
                - Pandas dataframe
                - Spark SQL
                - SQL database
                - Json file loading
                - CSV file loading
                - ...
            :param chat_handle Handle or reference to the large language model
            :param agent_name name of agent
            :param argument Parameter for agent or toolkit (pd.Dataframe for Pandas, Spark Dataframe
            :param streaming FLag to specify is streaming to stdout is enabled
        """
        if agent_name == LLMToolkitAgent.pandas_df_agent_name:
            from langchain.agents import create_pandas_dataframe_agent

            if argument.__class__.__name != 'pd.Dataframe':
                raise NotImplementedError(f'Agent {agent_name} should have argument of type pd.Dataframe')
            pandas_agent = create_pandas_dataframe_agent(llm=chat_handle, df=argument, verbose=True)
            return cls(chat_handle, pandas_agent)

        elif agent_name == LLMToolkitAgent.sql_agent_name:
            from langchain.agents.agent_toolkits import SQLDatabaseToolkit
            from langchain.agents import create_sql_agent

            sql_toolkit = SQLDatabaseToolkit(db=argument, llm=chat_handle)
            sql_agent = create_sql_agent(llm=chat_handle, toolkit=sql_toolkit, verbose=True)
            return cls(chat_handle, sql_agent)

        elif agent_name == LLMToolkitAgent.csv_agent_name:
            from langchain.agents import create_csv_agent

            csv_agent = create_csv_agent(chat_handle, argument, verbose=True)
            return cls(chat_handle, csv_agent)

        elif agent_name == LLMToolkitAgent.spark_sql_agent_name:
            from langchain.utilities.spark_sql import SparkSQL
            from langchain.agents import create_spark_sql_agent
            from langchain.agents.agent_toolkits import SparkSQLToolkit

            spark_sql = SparkSQL(schema=argument)
            spark_sql_agent = create_spark_sql_agent(
                llm=chat_handle,
                toolkit=SparkSQLToolkit(db=spark_sql, llm=chat_handle),
                verbose=True)
            return cls(chat_handle, spark_sql_agent)

        else:
            raise NotImplementedError(f'Agent {agent_name} is not supported')

    @staticmethod
    def load_from_json(argument: AnyStr) -> List[Dict[AnyStr, Any]]:
        import json

        filename = f'../../input/{argument}'
        chat_handle = ChatOpenAI(temperature=0)
        with open(filename) as f:
            content = f.read()
            json_array = content.split('},')
            result = [json.loads(entry) for entry in json_array]
            return result


if __name__ == '__main__':
    from test.domain.entities import Entities

    contractors_list = Entities.load('../../input/contractors.json')
    print(str(contractors_list))
    chat = ChatOpenAI(temperature=0)
    llm_json_agent = LLMToolkitAgent.build_from_toolkit(chat, 'json', '../input/contractors.json')
    answer1 = llm_json_agent.run("List all the contractors in San Jose")
    print(answer1)
"""
    tools = ['llm_langchain-math']
    from llm_langchain.chatgpttoolagent import ChatGPTToolAgent
    from langchain.agents import AgentType
    from langchain.tools.python.tool import PythonREPLTool

    chatGPTSimpleAgent = ChatGPTToolAgent.build(tools, AgentType.ZERO_SHOT_REACT_DESCRIPTION, True)
    chatGPTSimpleAgent.append_tool(PythonREPLTool())
    answer2 = chatGPTSimpleAgent.run("List all the contractors in San Jose")

    print(answer2)
  """
