"""
## Summarize and search financial documents using OpenAI's LLMs and Pinecone vector database

This DAG extracts and splits financial reporting data from the US 
[Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and 
ingests the data to a Pinecone vector database for generative question answering.  The DAG 
also creates and vectorizes summarizations of the 10-Q document using OpenAI completions.
"""
from __future__ import annotations

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.pinecone.hooks.pinecone import PineconeHook
from airflow.providers.openai.hooks.openai import OpenAIHook
from include.tasks import extract, split, summarize

import datetime
import logging
import pandas as pd
import uuid

PINECONE_CONN_ID = "pinecone_default"
OPENAI_CONN_ID = "openai_default"

logger = logging.getLogger("airflow.task")

edgar_headers={"User-Agent": "test1@test1.com"}

index_names = ["tenq", "tenq-summary"]

default_args = {"retries": 3, "retry_delay": 30, "trigger_rule": "none_failed"}


@dag(
    schedule_interval=None,
    start_date=datetime.datetime(2023, 9, 27),
    catchup=False,
    is_paused_upon_creation=True,
    default_args=default_args,
    params={
        "ticker": Param(
            default="",
            title="Ticker symbol from a US-listed public company.",
            type="string",
            description="US-listed companies can be found at https://www.sec.gov/file/company-tickers"
        )
    }
)
def FinSum_Pinecone(ticker: str = None):
    """
    This DAG extracts and splits financial reporting data from the US 
    [Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and 
    ingests the data to a Pinecone vector database for generative question answering.  The DAG 
    also creates and vectorizes summarizations of the 10-Q document.
    """

    def check_indexes() -> [str]:
        """
        Check if indexes exists.
        """

        pinecone_hook = PineconeHook(PINECONE_CONN_ID)
        
        if set(index_names).issubset(set(pinecone_hook.list_indexes())):
                return ["extract"]
        else:
            return ["create_indexes"]

    def create_indexes(existing: str = "ignore", pod_type:str = 's1'):

        pinecone_hook = PineconeHook(PINECONE_CONN_ID)
        
        for index_name in index_names:

            if index_name in pinecone_hook.list_indexes():
                if existing == "replace":
                    pinecone_hook.delete_index(index_name=index_name)
                elif existing == "ignore":
                    continue 
            else:
                pinecone_hook.create_index(
                    index_name=index_name, 
                    metric="cosine", 
                    replicas=1, 
                    dimension=1536, 
                    shards=1, 
                    pods=1, 
                    pod_type=pod_type, 
                    source_collection='',
                )

    def pinecone_ingest(df: pd.DataFrame, content_column_name: str, index_name: str):
        """
        This task concatenates multiple dataframes from upstream dynamic tasks and vectorizes 
        with import to pinecone.

        :param df: A dataframe from an upstream split task
        :param content_column_name: The name of the column with text to embed and ingest
        :param index_name: The name of the index to import data. 
        """

        openai_hook = OpenAIHook(OPENAI_CONN_ID)
        pinecone_hook = PineconeHook(PINECONE_CONN_ID)

        df["metadata"] = df.drop([content_column_name], axis=1).to_dict('records')

        df["id"] = df[content_column_name].apply(lambda x: str(
            uuid.uuid5(name=x+index_name, namespace=uuid.NAMESPACE_DNS)))

        df["values"] = df[content_column_name].apply(
            lambda x: openai_hook.create_embeddings(text=x, 
                                                    model="text-embedding-ada-002"))
        
        data = list(df[["id", "values", "metadata"]].itertuples(index=False, name=None))
        
        pinecone_hook.upsert_data_async(
            data=data,
            index_name=index_name, 
            async_req=True, 
            pool_threads=30,
            )
        
    _check_index = task.branch(check_indexes)()

    _create_index = task(create_indexes)()

    edgar_docs = task(extract.extract_10q)(ticker=ticker, edgar_headers=edgar_headers)

    split_docs = task(split.split_html)(df=edgar_docs)
    
    task(pinecone_ingest, task_id="import_chunks")(
        index_name=index_names[0], content_column_name="content", df=split_docs
        )

    generate_summary = task(summarize.summarize_openai)(df=split_docs, openai_conn_id=OPENAI_CONN_ID)

    task(pinecone_ingest, task_id="import_summary")(
        index_name=index_names[1], content_column_name="summary", df=generate_summary
        )
    
    _check_index >> _create_index >> edgar_docs

FinSum_Pinecone(ticker="")
