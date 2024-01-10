import os
import glob
import time
import pandas as pd
import pickle as pkl
import numpy as np
import transformers
import torch
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index import SQLDatabase, VectorStoreIndex, ServiceContext
from llama_index.indices.struct_store import (
    SQLTableRetrieverQueryEngine,
    NLSQLTableQueryEngine,
)
from llama_index.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from sqlalchemy import create_engine
import sqlite3
import warnings

warnings.filterwarnings("ignore")
import streamlit as st
from llama_index.llms import OpenAI


@st.cache_resource
def get_llm(model_name, token, cache_dir):
    if model_name.lower() == "openai":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        return llm

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=cache_dir
    )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=token,
        trust_remote_code=True,
        cache_dir=cache_dir,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=1500,
        generate_kwargs={"temperature": 0.00, "do_sample": False, "top_k": 100},
        tokenizer=tokenizer,
        model_name=model_name,
        device_map="cuda:0",
        model_kwargs={
            "trust_remote_code": True,
            "config": model_config,
            "quantization_config": bnb_config,
            "use_auth_token": token,
            "cache_dir": cache_dir,
        },
    )

    return llm


@st.cache_resource
def get_service_context(model_name, token, cache_dir):
    llm = get_llm(model_name, token, cache_dir)

    if model_name.lower() == "openai":
        service_context = ServiceContext.from_defaults(llm=llm)
    else:
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en"
        )

    return service_context


def populate_database_portfolio(conn, EXCEL_FILE_PATH, sheet="low risk"):
    conn.execute("DROP TABLE IF EXISTS portfolio")
    conn.commit()

    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df.to_sql(f"portfolio", con=conn, index=False, if_exists="replace")


def populate_database_history(conn, SOURCE_DOCUMENTS_PATH):
    """ "
    Populates the database with historical data.
    """

    # find all csv files in the csv folder
    all_csv_files = glob.glob(
        os.path.join(SOURCE_DOCUMENTS_PATH, "**/*.csv"), recursive=True
    )

    for file in all_csv_files:
        df = pd.read_csv(file)
        df = df.rename(columns={"company_ticker": "Ticker"})
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        ticker = file.split("/")[-1].split(".")[0]
        df.to_sql(
            "history_" + ticker.replace("-", "_"),
            con=conn,
            index=False,
            if_exists="replace",
        )

    # print("Database populated with History")


def populate_database_csv(conn, csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df.to_sql("asset_mapping", con=conn, index=False, if_exists="replace")

    # print("Database populated with Asset Mapping")


def delete_database(db):
    os.remove(db)
    # print("Database Deleted")


@st.cache_resource
def get_database(db, EXCEL_FILE_PATH, portfolio, SOURCE_DOCUMENTS_PATH, csv_path):
    conn = sqlite3.connect(db)
    engine = create_engine(f"sqlite:///{db}")

    if SOURCE_DOCUMENTS_PATH:
        populate_database_history(conn, SOURCE_DOCUMENTS_PATH=SOURCE_DOCUMENTS_PATH)

    if EXCEL_FILE_PATH:
        populate_database_portfolio(
            conn, EXCEL_FILE_PATH=EXCEL_FILE_PATH, sheet=portfolio
        )

    if csv_path:
        populate_database_csv(conn, csv_path=csv_path)

    sql_database = SQLDatabase(engine)
    return sql_database


def get_response(query_engine, query_str, print_=True):
    start = time.time()
    response = query_engine.query(query_str)
    end = time.time()

    if print_:
        print("-" * 60)
        print(f"TIME TAKEN: {end - start} secs")
        print(f"\nUSER QUERY:\n{query_str}")
        print(f"\nRESPONSE:\n{response.response}")
        print(f"\nSQL QUERY:\n{response.metadata['sql_query']}")
        print("-" * 60)

    return (end - start), response.response, response.metadata["sql_query"]


def get_query_engine(sql_database, service_context):
    table_node_mapping = SQLTableNodeMapping(sql_database)

    table_schema_objs = [
        SQLTableSchema(
            table_name=table,
            context_str=f"""
            This table contains daily stock prices of the ticker {table.split('_')[1]} for the last two years. 
            The columns in the table are: date, open, high, low, close, volume, dividends, stock_splits, ticker
            """,
        )
        for table in sql_database.get_usable_table_names()
        if table.startswith("history")
    ]

    table_schema_objs.extend(
        [
            SQLTableSchema(
                table_name=table,
                context_str=f"""
            This table is the 'portfolio'. The user might query this as 'user portfolio' Might also be referred to as portfolio.
            The columns in 'portfolio' table are:
                - type: type of asset (Stock/ETF/Cryptos/Gold)
                - date: date of transaction
                - ticker: company ticker 
                - shares_bought: number of shares bought for the Ticker
                - price_per_share: Price at which the respective shares were bought
                - total_cost: shares_bought * price_per_share
            """,
            )
            for table in sql_database.get_usable_table_names()
            if table.startswith("portfolio")
        ]
    )

    table_schema_objs.extend(
        [
            SQLTableSchema(
                table_name=table,
                context_str=f"""
            This table contains the asset mapping. You can relate company 'Ticker' to company 'Name'.
            The columns in this table are: ticker, name
            """,
            )
            for table in sql_database.get_usable_table_names()
            if table.startswith("asset_mapping")
        ]
    )

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
        service_context=service_context,
    )

    query_engine = SQLTableRetrieverQueryEngine(
        sql_database,
        obj_index.as_retriever(similarity_top_k=1),
        service_context=service_context,
    )

    return query_engine
