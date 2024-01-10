"""
## Summarize and search financial documents using Cohere's LLMs and the pgvector extensions of postgres.

This DAG extracts and splits financial reporting data from the US 
[Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and ingests 
the data to a PgVector vector database for generative question answering.  The DAG also 
creates and vectorizes summarizations of the 10-Q document.
"""
from __future__ import annotations

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.cohere.hooks.cohere import CohereHook
from airflow.providers.pgvector.hooks.pgvector import PgVectorHook
from include.tasks import extract, split, summarize

import datetime
import logging
import pandas as pd
import uuid

POSTGRES_CONN_ID = "postgres_default"
COHERE_CONN_ID = "cohere_default"

logger = logging.getLogger("airflow.task")

edgar_headers={"User-Agent": "test1@test1.com"}

table_names=["tenq", "tenq_summary"]

default_args = {"retries": 3, "retry_delay": 30, "trigger_rule": "none_failed"}


@dag(
    schedule_interval=None,
    start_date=datetime.datetime(2023, 9, 27),
    catchup=False,
    is_paused_upon_creation=True,
    default_args=default_args,
    params={
        "ticker": Param(
            "",
            title="Ticker symbol from a US-listed public company.",
            type="string",
            description="US-listed companies can be found at https://www.sec.gov/file/company-tickers"
        )
    }
)
def FinSum_PgVector(ticker: str = None):
    """
    This DAG extracts and splits financial reporting data from the US 
    [Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and ingests 
    the data to a PgVector vector database for generative question answering.  The DAG also 
    creates and vectorizes summarizations of the 10-Q document.
    """

    def check_tables() -> [str]:
        """
        Check if tables exists.
        """

        pgvector_hook = PgVectorHook(POSTGRES_CONN_ID)

        exists = []
        for table_name in table_names:
            if pgvector_hook.get_records(
                f"""SELECT * FROM pg_catalog.pg_tables
                    WHERE schemaname = 'public' 
                    AND tablename = '{table_name}';"""):
                exists.append(True)
            else:
                exists.append(False)
            
        if all(exists):
            return ["extract"]
        else:
            return ["create_tables"]

    def create_tables():

        pgvector_hook = PgVectorHook(POSTGRES_CONN_ID)

        pgvector_hook.create_extension('vector')

        for table_name in table_names:
            pgvector_hook.create_table(
                table_name=table_name,
                columns=[
                    "id UUID PRIMARY KEY",
                    "docLink TEXT",
                    "tickerSymbol TEXT",
                    "cikNumber TEXT",
                    "fiscalYear TEXT",
                    "fiscalPeriod TEXT",
                    "vector VECTOR(768)"
                ]   
            )

    def pgvector_ingest(df: pd.DataFrame, content_column_name: str, table_name: str):
        """
        This task concatenates multiple dataframes from upstream dynamic tasks and vectorizes 
        with import to a pgvector database.

        :param df: A dataframe from an upstream split task
        :param content_column_name: The name of the column with text to embed and ingest
        :param index_name: The name of the index to import data. 
        """

        pgvector_hook = PgVectorHook(POSTGRES_CONN_ID)
        cohere_hook = CohereHook(COHERE_CONN_ID)

        df["id"] = df[content_column_name].apply(
            lambda x: str(uuid.uuid5(
                name=x, namespace=uuid.NAMESPACE_DNS)
            )
        )

        df["vector"] = df[content_column_name].apply(
            lambda x: cohere_hook.create_embeddings(
                texts=[x], model="embed-multilingual-v2.0"
                )[0]
            )
        
        df.drop(content_column_name, axis=1).to_sql(
            name=table_name, 
            con=pgvector_hook.get_sqlalchemy_engine(), 
            if_exists='replace', 
            chunksize=1000
        )
        
    _check_index = task.branch(check_tables)()

    _create_index = task(create_tables)()

    edgar_docs = task(extract.extract_10q)(ticker=ticker, edgar_headers=edgar_headers)

    split_docs = task(split.split_html)(df=edgar_docs)

    task(pgvector_ingest, task_id="ingest_chunks")(
        table_name=table_names[0], content_column_name="content", df=split_docs)
    
    generate_summary = task(summarize.summarize_cohere)(df=split_docs, cohere_conn_id=COHERE_CONN_ID)

    task(pgvector_ingest, task_id="ingest_summaries")(
        table_name=table_names[1], content_column_name="summary", df=generate_summary)

    _check_index >> _create_index >> edgar_docs

FinSum_PgVector(ticker="")
