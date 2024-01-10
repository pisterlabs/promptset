import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

from sqlalchemy import insert
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index import Document, ListIndex
from llama_index import SQLDatabase, ServiceContext
from llama_index.llms import ChatMessage, OpenAI
from typing import List
import ast
import openai


DATA_PATH = "reddit_legal_cluster_test_results.parquet"

def clean_names(df):
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

 

def get_df():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df['datestamp'] = df['timestamp'].dt.date
    df['text_label'] = pd.Categorical(df['text_label'])
    df['topic_title'] = pd.Categorical(df['topic_title'])
    df.rename(columns={'body': 'reviews'}, inplace=True)
    df = clean_names(df)
    return df

df = get_df()

rows = df[['index', 'reviews', 'text_label', 'topic_title', 'token_count', 'llm_title', 'state', 'kmeans_label']]

st.dataframe(rows)

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create product reviews SQL table
table_name = "legal_questions"
legal_questions_table = Table(
    table_name,
    metadata_obj,
    Column("index", Integer(), primary_key=True),
    Column("reviews", String(240), nullable=False),
    Column("topic_title", String(80)),
    Column("llm_title", String(80)),
    Column("state", String(8), primary_key=True),
    Column("token_count", Integer),
    Column("kmeans_label", Integer),
)
metadata_obj.create_all(engine)

sql_database = SQLDatabase(engine, include_tables=["legal_questions"])

        

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)


def generate_questions(user_query: str) -> List[str]:
  system_message = '''
  You are given with Postgres table with the following columns.

  reviews, text_label, state, token_count.
  
  Your task is to decompose the given question into the following two questions.

  1. Question in natural language that needs to be asked to retrieve results from the table.
  2. Question that needs to be asked on the top of the result from the first question to provide the final answer.

  Example:

  Input:
 Summarize the theme of state with the highest average token count

  Output:
  1. Get the reviews of the state with the highest average token count
  2. Summarize the main theme of the state
  '''

  messages = [
      ChatMessage(role="system", content=system_message),
      ChatMessage(role="user", content=user_query),
  ]
  generated_questions = llm.chat(messages).message.content.split('\n')

  return generated_questions


# Create SQL Query Engine
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["legal_questions"],
    synthesize_response = False,
    service_context = service_context
)

def sql_rag(user_query: str) -> str:
  text_to_sql_query, rag_query = generate_questions(user_query)

  sql_response = sql_query_engine.query(text_to_sql_query)

  sql_response_list = ast.literal_eval(sql_response.metadata["sql_query"])

  text = [' '.join(t) for t in sql_response_list]
  text = ' '.join(text)

  listindex = ListIndex([Document(text=text)])
  list_query_engine = listindex.as_query_engine()

  summary = list_query_engine.query(rag_query)

  return sql_response_list


if prompt := st.chat_input(placeholder="Send a message"):
    st.chat_message("user", avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/user_question_resized_pil.jpg").write(prompt)
    
    response = sql_rag(prompt)
    st.write(response)
    
    