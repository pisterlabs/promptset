import os
import sqlite3
import pandas as pd
# from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import retry
import logging

logging.basicConfig(level=logging.ERROR)

import streamlit as st

from globals import (
    DB_FILE, OPENAI_MODELS_COMPLETIONS, 
    DEFAULT_MODEL_CONFIG, LANG_MODEL_PRICING
)
from app_state import (state, init_app_state, _set_state_cb)
init_app_state() # ensure all state variables are initialized

# DATA -------------------------------------------------------------------------

@st.cache_data(persist='disk')
def csv_to_df(excel_file):
    df = pd.read_csv(excel_file)
    return df

@st.cache_data(persist='disk')
def excel_to_df(excel_file):
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
    # New in Pandas version 1.3.0.
    #   The engine xlrd now only supports old-style .xls files. When engine=None, the following logic will be used to determine the engine:
    #   If path_or_buffer is an OpenDocument format (.odf, .ods, .odt), then odf will be used.
    #   Otherwise if path_or_buffer is an xls format, xlrd will be used.
    #   Otherwise if path_or_buffer is in xlsb format, pyxlsb will be used.
    #   Otherwise openpyxl will be used.
    #
    # import openpyxl
    # df = pd.read_excel(excel_file, engine=openpyxl)
    #
    # Therefore... do not need to provide "engine" when using a "path_or_buffer"
    df = pd.read_excel(excel_file, engine='openpyxl')
    return df

def prepare_data(df):
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

@st.cache_resource()
def db_connection():
    return sqlite3.connect(DB_FILE , check_same_thread=False)

@st.cache_resource()
def sql_database(table):
    # create db engine
    # eng = create_engine(
    #     url=f'sqlite:///file:{DB_FILE}&cache=shared',
    #     poolclass=StaticPool, # single connection for requests
    #     creator=lambda: db_connection(),
    # )
    # db = SQLDatabase(engine=eng)

    db = SQLDatabase.from_uri(
        database_uri = f'sqlite:///file:{DB_FILE}&cache=shared',
        include_tables=[table],         # we include only one table to save tokens in the prompt :)
        sample_rows_in_table_info=2,    # we only need 2 rows to get the table info
        engine_args={'poolclass': StaticPool, 'creator': lambda: db_connection()},
    )
    return db

# OPENAI DATA QUERY ------------------------------------------------------------

# create OpenAI LLM connection
# NOTE: relies on environment key in case you want to
# remove entering the key in the app
def get_llm(
    model_name: str = DEFAULT_MODEL_CONFIG['completions_model'],
    temperature: float = DEFAULT_MODEL_CONFIG['temperature'],
    top_p: float = DEFAULT_MODEL_CONFIG['top_p'],
    max_tokens: int = DEFAULT_MODEL_CONFIG['max_tokens'],
    max_retries: int = 3,
    streaming: bool = False,
):
    return OpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'], 
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_retries=max_retries,
        streaming=streaming,
    )

@retry.retry(tries=2, delay=5, backoff=3, jitter=(1, 5), max_delay=60, logger=logging.getLogger("LLM DATA QUERY (get_llm_data_query_response)"))
def get_llm_data_query_response(query, table, model_name=DEFAULT_MODEL_CONFIG['completions_model'], intermediate_steps=False, limit=3):
    model_config = {
        'model_name': model_name,
        'temperature': 0,      # override settings = do not halucinate!
        'top_p': state.top_p,
        'max_tokens': 2000,    # override settings
    }
    llm = get_llm(**model_config)
    
    # create SQLDatabaseChain LLM connection
    db_chain = SQLDatabaseChain.from_llm(
        llm=llm, db=sql_database(table), verbose=True,
        # use_query_checker=True,
        return_intermediate_steps=intermediate_steps,
        top_k=limit
    )
    
    # run query and display result
    with get_openai_callback() as token_counter:
        if query:
            if state.intermediate_steps: 
                result = db_chain(query)
            else:
                result = db_chain.run(query)

    print('---- Data SQL Query ----', '\n',
          'LLM Prompt Tokens:', token_counter.prompt_tokens, '\n',
          'LLM Completion Tokens:', token_counter.completion_tokens, '\n',
          'Total LLM Token Count:', token_counter.total_tokens)

    estimated_cost = ((token_counter.prompt_tokens / 1000.0) * LANG_MODEL_PRICING[state.completions_model]['input']) + \
        ((token_counter.completion_tokens / 1000.0) * LANG_MODEL_PRICING[state.completions_model]['output'])
    print('Data SQL Query Estimated Cost: $', estimated_cost)
    state.estimated_cost_data = estimated_cost
    state.cumulative_cost += estimated_cost

    return result

# DATA CHAT PAGE ----------------------------------------------------------------

def main(title):
    # Sidebar
    with st.sidebar:
        st.markdown(f'#### {title} Settings')
        st.selectbox(
            'OpenAI model', options=OPENAI_MODELS_COMPLETIONS,
            on_change=_set_state_cb, kwargs={'completions_model': 'selectbox_data_completions_model_name'},
            index=OPENAI_MODELS_COMPLETIONS.index(state.completions_model),
            help='Allowed models. Accuracy, speed, token consumption and costs will vary.',
            key='selectbox_data_completions_model_name'
        )
        # results limit
        st.number_input(
            'Results limit', value=state.limit, min_value=1, max_value=10, step=1,
            on_change=_set_state_cb, kwargs={'limit': 'number_input_limit'},
            help='Limit the number of results returned, which can improve performance and save OpenAI costs',
            key='number_input_limit'
        )
    
    # Body
    st.subheader('Upload Data')
    excel_file = st.file_uploader('Choose an Excel file on your computer', type=['xlsx', 'csv'], accept_multiple_files=False)
    if excel_file is None:
        return
        
    if excel_file.type in ['application/vnd.ms-excel', 'application/octet-stream', 'text/csv']:
        df = csv_to_df(excel_file)
        # state.db_table = excel_file.name.replace('.csv', '').replace(' ', '_').lower()
    else: # 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        df = excel_to_df(excel_file)
        # state.db_table = excel_file.name.replace('.xlsx', '').replace(' ', '_').lower()

    if st.checkbox('Show Data', value=False):
        st.dataframe(df)

    # commit data to sql
    data = prepare_data(df)
    data.to_sql(state.db_table, db_connection(), if_exists='replace', index=False)

    st.subheader('Query Data')
    with st.form(key='data_chat_form', clear_on_submit=False):
        # user query
        st.text_input(
            'Enter a data query in plain English', value=state.query,
            help='Enter a question based on the uploaded dataset. Add as much detail as you like. '
            'E.g., "What is X of Y in the table. Limit to 10 results, and format as JSON showing X and Y values only."',
            key='text_input_query_data'
        )
        st.checkbox(
            'Show Intermediate Steps', value=state.intermediate_steps, 
            key='checkbox_intermediate_steps'
        )
        apply_query = st.form_submit_button(
            label='Ask', type='primary',
            on_click=_set_state_cb, kwargs={
                'intermediate_steps': 'checkbox_intermediate_steps',
                'query': 'text_input_query_data',
                'estimated_cost_data': 'estimated_cost_reset',
            },
        )

    if apply_query and state.query and state.openai_api_key:
        query = state.query + f' Strictly use only these data columns "{list(data.columns)}". ' + \
            'Do not wrap the SQL statement in quotes. Do not embelish the answer with any additional text.'
        result = get_llm_data_query_response(
            query, state.db_table,
            model_name=state.completions_model,
            intermediate_steps=state.intermediate_steps, 
            limit=state.limit
        )
        if state.intermediate_steps:
            with st.expander('Intermediate Steps', expanded=False):
                st.write(state.completions_model)
                st.write(result['intermediate_steps'])
            st.text(result['result'])
        else:
            st.text(result)
    elif apply_query and not state.query:
        st.info('Please enter a query above.')