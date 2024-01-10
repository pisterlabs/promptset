#!/usr/bin/env python
# coding: utf-8


import time
import pandas as pd
import gradio as gr

from typing import Any, Mapping, List, Dict, Optional, Tuple
from pydantic import BaseModel, Extra, root_validator

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import Generation, LLMResult
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult, HumanMessage, SystemMessage
from langchain.llms import VertexAI
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

from google.cloud import aiplatform


# @title Specify Project details and location of the BQ table

project_id = "<Project ID>"  # @param {type:"string"}
location = "<BigQuery region>"  # @param {type:"string"}
dataset_id = '<BQ dataset ID>' # @param {type:"string"}
table_name1 = 'country' # @param {type:"string"}
table_name2 = 'league' # @param {type:"string"}
table_name3 = 'player' # @param {type:"string"}
table_name4 = 'team' # @param {type:"string"}
table_name5 = 'match' # @param {type:"string"}

table_names = (table_name1, table_name2, table_name3, table_name4, table_name5)

table_uri = f"bigquery://{project_id}/{dataset_id}"
bq_engine = create_engine(f"bigquery://{project_id}/{dataset_id}")

query=f"""select * from {project_id}.{dataset_id}.{table_name1}"""
result = bq_engine.execute(query)
for row in result:
    print(row)
    
# query=f"""select * from {project_id}.{dataset_id}.{table_name2}"""
# result = bq_engine.execute(query).first()
# print(result)

# query=f"""select * from {project_id}.{dataset_id}.{table_name3}"""
# result = bq_engine.execute(query).first()
# print(result)

# query=f"""select * from {project_id}.{dataset_id}.{table_name4}"""
# result = bq_engine.execute(query).first()
# print(result)

# query=f"""select * from {project_id}.{dataset_id}.{table_name5}"""
# result = bq_engine.execute(query).first()
# print(result)

# Text model
llm = VertexAI(
    model_name="text-bison-32k",
    max_output_tokens=1024,
    temperature=0.0,
    top_p=0.8,
    top_k=40,
    verbose=True,
)


def bq_talker(question):
    # create SQLDatabase instance from BQ Engine
    bq_db = SQLDatabase(engine=bq_engine, metadata=MetaData(bind=bq_engine), include_tables=[x for x in table_names])
    
    # create SQL DB Chain with the initialized LLM and above SQLDB instance
    bq_db_chain = SQLDatabaseChain.from_llm(llm, bq_db, verbose=True, return_intermediate_steps=True)
    
    # Define prompt for BigQuery SQL
    _googlesql_prompt = """You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer to the input question.
      Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
      Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
      Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
      Use the following format:
      Question: "Question here"
      SQLQuery: "SQL Query to run"
      SQLResult: "Result of the SQLQuery"
      Answer: "Final answer here"
      Only use the following tables:
      {table_info}
      
      Please avoid to use like or any other wild char to match the row or data, it's better to user equal
      
      If someone asks for aggregation on a STRING data type column, then CAST column as NUMERIC before you do the aggregation.

      If someone asks for specific month, use ActivityDate between current month's start date and current month's end date

      If someone asks for column names in the table, use the following format:
      select column_name from '{project_id}.{dataset_id}'.INFORMATION_SCHEMA.COLUMNS
      where table_name in {project_id}.{dataset_id}.{table_info}
      
      Question: {input}
    
    """
    
    GoogleSql_Prompt = PromptTemplate(
        input_variables=["input", "table_info", "top_k", "project_id", "dataset_id"],
        template=_googlesql_prompt
    )
    
    # passing question to the prompt template
    final_prompt = GoogleSql_Prompt.format(input=question, project_id=project_id, dataset_id=dataset_id, table_info=table_names, top_k=10000)
    
    # pass final prompt to SQL Chain
    output = bq_db_chain(final_prompt)
    
    return output['result'], output['intermediate_steps'][1]

#Please use equal firstly to allocate the data, if there is nothing return, please use like or any other wild char to match


# bq_talker('what is short name for FC Barcelona')

# bq_talker('what BAR is')

# bq_talker('how many matchs FC Barcelona won in the 2008/2009 season as home team')

# bq_talker('here is the rule for each match, win = 3 points, draw = 1 point, lost = 0 point. how many points FC Barcelona had for season 2008/2009')

with gr.Blocks() as ui:
    gr.Markdown(
    """
    ## Ask BiqQuery

    This demo is to showcase answering questions on a tabular data available in Big Query using Vertex PALM LLM & Langchain.

    This demo uses a sample public dataset from Kaggle (https://www.kaggle.com/datasets/hugomathien/soccer)

    ### Sample Inputs:
    1. what is short name for FC Barcelona ?
    2. what BAR is ?
    3. how many matchs FC Barcelona won in the 2008/2009 season as home team ?
    4. here is the rule for each match, win = 3 points, draw = 1 point, lost = 0 point. how many points FC Barcelona had for season 2008/2009

    ### Enter a search query...

    """)
    
    with gr.Row():
        with gr.Column():
            input_text=gr.Textbox(label="Question", placeholder="how many matchs FC Barcelona won in the 2008/2009 season as home team")
            
    with gr.Row():
      generate = gr.Button("Talk to BigQuery")

    with gr.Row():
      lbl_output = gr.Textbox(label="Output")
    
    with gr.Row():
      lbl_sqlscript = gr.Textbox(label="SQL query generated by LLM")
    
    generate.click(bq_talker, input_text, [lbl_output, lbl_sqlscript])

# ui.launch(share=True, debug=False)
# ui.launch(server_name="127.0.0.1", server_port=8080, share=False, debug=False)
ui.launch(server_name="127.0.0.1", server_port=8080, debug=True)