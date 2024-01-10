"""
## Summarize and search financial documents using Cohere's LLMs.

This DAG extracts and splits financial reporting data from the US 
[Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and generates 
vector embeddings with cohere embeddings for generative question answering.  The DAG also 
creates and vectorizes summarizations of the 10-Q document using Cohere Summarize.
"""
from __future__ import annotations

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.cohere.hooks.cohere import CohereHook
from include.tasks import extract, split, summarize

import datetime
import logging
import pandas as pd
from pathlib import Path
import uuid

COHERE_CONN_ID = "cohere_default"

logger = logging.getLogger("airflow.task")

edgar_headers={"User-Agent": "test1@test1.com"}

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
def FinSum_Cohere(ticker: str = None):
    """
    This DAG extracts and splits financial reporting data from the US 
    [Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and generates 
    vector embeddings with cohere embeddings for generative question answering.  The DAG also 
    creates and vectorizes summarizations of the 10-Q document.

    With very large datasets it may not be convenient to store embeddings in a vector database.  This DAG
    shows how to save documents with vectors on disk. Realistically these would be serialized in cloud object 
    storage but for the purpose of demo we store them on local disk.

    """

    def vectorize(df: pd.DataFrame, content_column_name: str, output_file_name: Path) -> str:
        """
        This task concatenates multiple dataframes from upstream dynamic tasks and vectorizes 
        with Cohere Embeddings.  The vectorized dataset is written to disk.

        :param df: A dataframe from an upstream split task
        :param content_column_name: The name of the column with text to embed and ingest
        :param output_file_name: Path for saving embeddings
        :return: Location of saved file
        """

        cohere_hook = CohereHook(COHERE_CONN_ID)

        df["id"] = df[content_column_name].apply(
            lambda x: str(uuid.uuid5(
                name=x, 
                namespace=uuid.NAMESPACE_DNS))
            )

        df["vector"] = df[content_column_name].apply(
            lambda x: cohere_hook.create_embeddings(
                texts=[x], model="embed-multilingual-v2.0"
                )[0]
            )
        
        df.to_parquet(output_file_name)

        return output_file_name


    edgar_docs = task(extract.extract_10q)(ticker=ticker, edgar_headers=edgar_headers)

    split_docs = task(split.split_html)(df=edgar_docs)

    embeddings_file = task(vectorize)(
        output_file_name="include/data/html/cohere_embeddings.parquet",
        content_column_name="content",
        df=split_docs)
    
    generate_summary = task(summarize.summarize_cohere)(df=split_docs, cohere_conn_id=COHERE_CONN_ID)

    summaries_file = (
        task(vectorize, task_id="vectorize_summaries")(
            output_file_name="include/data/html/cohere_summary_embeddings.parquet",
            content_column_name="summary",
            df=generate_summary)
    )

FinSum_Cohere(ticker="")