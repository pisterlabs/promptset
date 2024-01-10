from __future__ import annotations

from airflow.providers.openai.hooks.openai import OpenAIHook
from airflow.providers.cohere.hooks.cohere import CohereHook
from cohere.client import Client as CohereClient
import logging
import openai as openai_client
import pandas as pd

logger = logging.getLogger("airflow.task")


def chunk_summarization_openai(
        openai_client: openai_client, content: str, ticker: str, fy: str, fp: str) -> str:
    """
    This function uses openai gpt-3.5-turbo-1106 to summarize a chunk of text.

    :param content: The text content to be summarized.
    :param ticker: The company ticker symbol for (status printing).
    :param fy: The fiscal year of the document chunk for (status printing).
    :param fp: The fiscal period of the document chunk for (status printing).
    :return: A summary string
    """

    logger.info(f"Summarizing chunk for ticker {ticker} {fy}:{fp}")
    
    response = openai_client.ChatCompletion().create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0,
        max_tokens=1000
        )
    if content:=response.get("choices")[0].get("message").get("content"):
        return content
    else:
        return None

def doc_summarization_openai(
        openai_client: openai_client, content: str, doc_link: str) -> str:
    """
    This function uses openai gpt-4-1106-preview to summarize a concatenation of
    document chunks.

    :param content: The text content to be summarized.
    :param doc_link: The URL of the document being summarized (status printing).
    :return: A summary string
    """

    logger.info(f"Summarizing document for {doc_link}")

    response = openai_client.ChatCompletion().create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0,
        max_tokens=1000
        )
    if content:=response.get("choices")[0].get("message").get("content"):
        return content
    else:
        return None
    
def summarize_openai(df: pd.DataFrame, openai_conn_id:str) -> pd.DataFrame:
    """
    This task uses openai to recursively summarize extracted documents. First the individual
    chunks of the document are summarized.  Then the collection of chunk summaries are summarized.

    :param df: A Pandas dataframe from upstream split tasks
    :param openai_conn_id: The connection name to use for the openai hook.
    :return: A Pandas dataframe with summaries for ingest to a vector DB.
    """
    
    openai_client.api_key = OpenAIHook(openai_conn_id)._get_api_key()

    df["chunk_summary"] = df.apply(lambda x: chunk_summarization_openai(
        openai_client=openai_client, 
        content=x.content, 
        fy=x.fiscalYear, 
        fp=x.fiscalPeriod, 
        ticker=x.tickerSymbol), axis=1)

    summaries_df = df.groupby("docLink").chunk_summary.apply("\n".join).reset_index()

    summaries_df["summary"] = summaries_df.apply(lambda x: doc_summarization_openai(
        openai_client=openai_client, 
        content=x.chunk_summary, 
        doc_link=x.docLink), axis=1)
    
    summaries_df.drop("chunk_summary", axis=1, inplace=True)

    summary_df = df.drop(["content", "chunk_summary"], axis=1).drop_duplicates().merge(summaries_df)

    return summary_df

def chunk_summarization_cohere(
        cohere_client: CohereClient, content: str, ticker: str, fy: str, fp: str) -> str:
    """
    This function uses Cohere's "Summarize" endpoint to summarize a chunk of text.

    :param content: The text content to be summarized.
    :param ticker: The company ticker symbol for (status printing).
    :param fy: The fiscal year of the document chunk for (status printing).
    :param fp: The fiscal period of the document chunk for (status printing).
    :return: A summary string
    """

    logger.info(f"Summarizing chunk for ticker {ticker} {fy}:{fp}")
    
    return cohere_client.summarize(
        text=content,
        model="command",
        length="long",
        extractiveness="medium",
        temperature=1,
        format="paragraph"
    ).summary

def doc_summarization_cohere(
        cohere_client: CohereClient, content: str, doc_link: str) -> str:
    """
    This function uses Cohere's "Summarize" endpoint to summarize a concatenation 
    of chunk summaries.

    :param content: The text content to be summarized.
    :param doc_link: The URL of the document being summarized (status printing).
    :return: A summary string
    """

    logger.info(f"Summarizing document for {doc_link}")

    return cohere_client.summarize(
        text=content,
        model="command",
        length="long",
        extractiveness="medium",
        temperature=1,
        format="paragraph"
    ).summary
    
def summarize_cohere(df: pd.DataFrame, cohere_conn_id:str) -> pd.DataFrame:
    """
    This task uses cohere to recursively summarize extracted documents. First the individual
    chunks of the document are summarized.  Then the collection of chunk summaries are summarized.

    :param df: A Pandas dataframe from upstream split tasks
    :param cohere_conn_id: An Airflow connection ID for Cohere
    :return: A Pandas dataframe with summaries for ingest to a vector DB.
    """

    cohere_client = CohereHook(cohere_conn_id).get_conn

    df["chunk_summary"] = df.apply(lambda x: chunk_summarization_cohere(
        cohere_client=cohere_client,
        content=x.content, 
        fy=x.fiscalYear, 
        fp=x.fiscalPeriod, 
        ticker=x.tickerSymbol), axis=1)

    summaries_df = df.groupby("docLink").chunk_summary.apply("\n".join).reset_index()

    summaries_df["summary"] = summaries_df.apply(lambda x: doc_summarization_cohere(
        cohere_client=cohere_client,
        content=x.chunk_summary, 
        doc_link=x.docLink), axis=1)
    
    summaries_df.drop("chunk_summary", axis=1, inplace=True)

    summary_df = df.drop(["content", "chunk_summary"], axis=1).drop_duplicates().merge(summaries_df)

    return summary_df
