#!/usr/bin/env python
# coding: utf-8

import ast
import time
import json
import gradio as gr
from typing import Any, Mapping, List, Dict, Optional, Tuple
from pydantic import BaseModel, Extra, root_validator
import pandas as pd
import pandas_gbq
import plotly.express as px

from google.cloud import bigquery
from google.cloud import storage

from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
# Create chain to answer questions
from langchain.chains import RetrievalQA

from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
from langchain.llms.utils import enforce_stop_tokens
from langchain.schema import Generation, LLMResult
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult, HumanMessage, SystemMessage
from langchain.llms import VertexAI
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate


# @title Specify Project details and location of the BQ table

project_id = "hello-world-360207"  # @param {type:"string"}
location = "us-central1"  # @param {type:"string"}
dataset_id = 'demo_talk2bq' # @param {type:"string"}
table_name1 = 'country' # @param {type:"string"}
table_name2 = 'league' # @param {type:"string"}
table_name3 = 'player' # @param {type:"string"}
table_name4 = 'team' # @param {type:"string"}
table_name5 = 'match' # @param {type:"string"}

table_names = (table_name1, table_name2, table_name3, table_name4, table_name5)

#setup bigquery engine
table_uri = f"bigquery://{project_id}/{dataset_id}"
bq_engine = create_engine(f"bigquery://{project_id}/{dataset_id}")


# Text model
llm = VertexAI(
    model_name="text-bison",
    max_output_tokens=512,
    temperature=0.0,
    top_p=0.8,
    top_k=40,
    verbose=True,
)


# convert sql result from bq_talker to pandas dataframe format
def convertToDataframe(strDataFrame):
    # 将string转换成list
    list_of_tuples = ast.literal_eval(strDataFrame)

    # 打印list
    # print(list_of_tuples)

    # print(type(list_of_tuples))
    
    df = pd.DataFrame.from_dict(list_of_tuples)
    
    return df


def showBarPlot(strDataFrame):
    
    df = convertToDataframe(strDataFrame)
    df = df.rename(columns={0: 'Club', 1: 'Goals'})
    
    return gr.BarPlot.update(
        df,
        x="Club",
        y="Goals",
        color="Goals",
        title="2008/2009 season, top 10 teams in terms of goals scored",
        tooltip=["Club", "Goals"],
        width = 800,
        # y_lim=[20, 100],
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
      
      Please use equal first to filter and locate data. When you cannot find the data you need through equal, use like and other wild char.
      
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
    
    return output['result'], output['intermediate_steps'][1], output['intermediate_steps'][3]
    #return output['result'], output['intermediate_steps'][1]

#Please use equal firstly to allocate the data, if there is nothing return, please use like or any other wild char to match


# bq_talker('what is short name for FC Barcelona')

# result, sql_script, df_str = bq_talker('What is short name for FC Barcelona')

# result, sql_script, df_str = bq_talker('In the 2008/2009 season, who are the top 10 teams in terms of goals scored?')

# bq_talker('In the 2008/2009 season, who are the top 10 teams in terms of goals scored, and how many goals they scored at home and away?')

# bq_talker('how many matchs FC Barcelona won in the 2008/2009 season as home team')

# bq_talker('here is the rule for each match, win = 3 points, draw = 1 point, lost = 0 point. how many points FC Barcelona had for season 2008/2009')


with gr.Blocks() as ui:
    gr.Markdown(
    """
    ## Ask BiqQuery

    This demo is to showcase answering questions on a tabular data available in Big Query using Vertex PALM LLM & Langchain.

    This demo uses a sample public dataset from Kaggle (https://www.kaggle.com/datasets/hugomathien/soccer)

    ### Sample Inputs:
    1. What is short name for FC Barcelona ?
    2. In the 2008/2009 season, who are the top 10 teams in terms of goals scored?
    3. In the 2008/2009 season, who are the top 10 teams in terms of goals scored, and how many goals they scored at home and away?
    4. How many matches FC Barcelona won in the 2008/2009 season as home team ?
    5. Here is the rule for each match, win = 3 points, draw = 1 point, lost = 0 point. how many points FC Barcelona had for season 2008/2009

    ### Enter a search query...

    """)
    
    with gr.Row():
        with gr.Column():
            input_text=gr.Textbox(label="Question", placeholder="how many matchs FC Barcelona won in the 2008/2009 season as home team")
            
    with gr.Row():
      btnTalk2bq = gr.Button("Talk to BigQuery")

    with gr.Row():
      lbl_output = gr.Textbox(label="Output")
    
    with gr.Row():
      lbl_sqlscript = gr.Textbox(label="SQL query generated by LLM")
    
    with gr.Row():
      lbl_sqlresult = gr.Textbox(label="Data fetched by SQL scipt that generated by LLM")
            
    with gr.Row():
      btnShowBarPlot = gr.Button("Show Bar Plot")
        
    with gr.Row():
      plot = gr.BarPlot()
    
    btnShowBarPlot.click(showBarPlot, lbl_sqlresult, outputs=plot)
    
    #display.change(bar_plot_fn, inputs=display, outputs=plot)
    
    #btnTalk2bq.click(bq_talker, input_text, [lbl_output, lbl_sqlscript])
    btnTalk2bq.click(bq_talker, input_text, [lbl_output, lbl_sqlscript, lbl_sqlresult])

#ui.launch(server_name="0.0.0.0", server_port=1234, share=True, debug=True)
ui.launch(server_name="0.0.0.0", server_port=8080, debug=True)
