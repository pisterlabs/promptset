import streamlit as st
import os
from langchain.llms import VertexAI #LlamaCpp
from google.cloud import bigquery
import logging
import pandas
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate


project = os.environ['GOOGLE_CLOUD_PROJECT']
dataset = os.environ['BILLING_DATASET']
table = os.environ['BILLING_TABLE']
sqlalchemy_url = f"bigquery://{os.environ['GOOGLE_CLOUD_PROJECT']}/{dataset}"


def process_prompt():
    user_query = st.session_state.user_query
    schema = ""
    with open("schema.json", "r") as f:
        schema = f.read()

    TEMPLATE = """Only use the following tables:
    {table}.
    The schema of the table is: {schema}

    If accessing sub keys of a field, you must use UNNEST to flatten the data.

    where service.id is an unusable identifier and service.description is the name of the service
    use invoice.month for any data prior to the current month
    INTERVAL must be in days, not months or years

    You can not query nested fields, so e.g. SELECT `gcp_billing_export_v1_010767_AD0D5D_BCC8F6`.`project`.`number` is not a valid query
    
    Some examples of SQL queries that corrsespond to questions are:

    input: how much did I spend on compute in the last 90 days?
    output: SELECT
        sum(total_cost) as my_cost,
        FORMAT_DATE("%Y-%m", usage_start_time) AS month,
    FROM `{dataset}`
        WHERE service.description LIKE "%Compute%"
        AND usage_start_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 90 day
    GROUP BY month
    
    input: how much did I spend in the last month?
    output: SELECT
        sum(total_cost) as my_cost,
        FORMAT_DATE("%Y-%m", usage_start_time) AS month
    FROM `{dataset}`
        WHERE usage_start_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 30 day
    GROUP BY month

    input: how much did I spend last month?
    output: SELECT
        sum(total_cost) as my_cost,
        FORMAT_DATE("%Y-%m", usage_start_time) AS month
    FROM `{dataset}`
        WHERE usage_start_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 60 day
        AND usage_start_time <= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 30 day
    GROUP BY month

    input: how much did I spend on compute over the last 6 months?
    output: SELECT
        sum(cost) as my_cost,
        FORMAT_DATE("%Y-%m", usage_start_time) AS month
    FROM `{dataset}`
    WHERE service.description LIKE "%Compute Engine%" 
    AND usage_start_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 180 day

    input: How much did I spend on vertex in the last month?
    output: SELECT SUM(cost) AS total_cost 
    FROM `{dataset}`
    WHERE service.description LIKE "%Vertex% 
    AND usage_start_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 30 day

    input: How much did I spend on BQ over the last 6 months?
    output: SELECT SUM(cost) AS total_cost,
    FORMAT_DATE("%Y-%m", usage_start_time) AS month
    FROM `{dataset}`
    WHERE service.description LIKE "%BigQuery%"
    AND usage_start_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) - INTERVAL 180 day

    input: Write a BigQuery SQL query for: {user_query}
    output:"""

    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["schema", "user_query", "dataset", "table"], template=TEMPLATE
    )

    llm = VertexAI(model_name="code-bison", max_output_tokens=2048)
    db = SQLDatabase.from_uri(sqlalchemy_url)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    top_k=100,
    handle_parsing_errors=True,
    )


    # in case it decides to spit out markdown JSON
    out = agent_executor.run(CUSTOM_PROMPT)
    st.write("LLM said: {}".format(out))
    # st.write("Running query... \n```\n{}\n```".format(sql))
    # out = run_query(sql)
    st.bar_chart(out)

    # st.write(out)

user_query = st.chat_input("Ask me a question about your bill", on_submit=process_prompt, key="user_query")
