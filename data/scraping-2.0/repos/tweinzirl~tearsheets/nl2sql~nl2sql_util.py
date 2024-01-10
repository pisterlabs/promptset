from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
    
import pandas as pd
import os
import openai
from sqlalchemy import exc, create_engine, text as sql_text
import argparse
import gradio as gr

Message_Template_Filename = 'nl2sql/Template_MySQL-1.txt'
VDSDB_Filename =  "nl2sql/Question_Query_Embeddings-1.txt"
VDSDB = "Dataframe"
LLM_MODEL = 'gpt-3.5-turbo'

# for local modules
from nl2sql.NL2SQL_functions import Prepare_Message_Template, Run_Query
from nl2sql.lib_OpenAI_Embeddings import VDS


MYSQL_USER = os.getenv("MYSQL_USER", None)
MYSQL_PWD = os.getenv("MYSQL_PWD", None)
openai.api_key = OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
MYSQL_Credentials = {'User':MYSQL_USER,'PWD':MYSQL_PWD}

Embedding_Model = "text-embedding-ada-002"
Encoding_Base = "cl100k_base"
Max_Tokens = 250
Temperature = 0
Token_Cost = {"gpt-3.5-turbo-instruct": {"Input":0.0015/1000,"Output":0.002/1000},
                LLM_MODEL: {"Input":0.001/1000,"Output":0.002/1000},
                "text-embedding-ada-002": {"Input":0.0001/1000, "Output":0.0001/1000}}

LOCAL_VDS = VDS(VDSDB_Filename, Encoding_Base, Embedding_Model, Token_Cost, Max_Tokens) 
LOCAL_VDS.Load_VDS_DF(Verbose=False)

def sql_to_df(Message, return_sql=False, Verbose=False, Debug=False):
    '''
    Given a natural language query, builds a sql query, executes against the
    database, and returns the result as a dataframe.

    If return_sql is True, the sql query is also returned, so the output is a
    tuple (sql_query, dataframe). Otherwise, only the dataframe is returned.
    '''

    message_history = Prepare_Message_Template(Template_Filename = Message_Template_Filename, Verbose=False, Debug=False)

    Question_Emb = LOCAL_VDS.OpenAI_Get_Embedding(Text=Message, Verbose=Verbose)
    rtn = LOCAL_VDS.Search_VDS(Question_Emb, Similarity_Func = 'Cosine', Top_n=3)

    N_Shot_Examples = {'Question':rtn[1], 'Query':rtn[2]}
    # Append N Shot Examples to Message_History
    for i in range(len(N_Shot_Examples['Question'])):
        message_history.append({"role": "system", "name":"example_user", "content": N_Shot_Examples['Question'][i]})
        message_history.append({"role": "system", "name":"example_assistant", "content": N_Shot_Examples['Query'][i]})

    # Append Message (e.g. question)
    message_history.append({"role": "user", "content": Message})

    if Debug:
        print(f'{message_history}')
    
    # pass to LLM    
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=message_history,
        temperature=0
      #  stream=True
    )

    Response = response['choices'][0]['message']['content']
    message_history.append({'role': 'assistant', 'content': Response})

    # now need to query DB
    df = Run_Query(Credentials=MYSQL_Credentials, Query=Response)

    if return_sql:
        return Response, df
    else:
        return df
