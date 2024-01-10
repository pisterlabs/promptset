from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.schema import SystemMessage

from tools import execute_sql,get_table_columns,get_table_column_distr
from visualization_tools import create_redash_query, create_redash_visualization, create_redash_dashboard, add_widget_on_dashboard, publish_dashboard

with open("system_message.txt", "r") as file:
    system_message = file.read()


def get_agent_executor():
    agent_kwargs = {
    "system_message": SystemMessage(content=system_message)
    }

    analyst_agent_openai = initialize_agent(
        llm=ChatOpenAI(temperature=0.1, model = 'gpt-4-1106-preview'),
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=[execute_sql, get_table_columns, get_table_column_distr, create_redash_query, create_redash_visualization, create_redash_dashboard, add_widget_on_dashboard, publish_dashboard],
        agent_kwargs=agent_kwargs,
        verbose=True,
        max_iterations=20,
        early_stopping_method='generate'
    )

    return analyst_agent_openai
