"""
## Summarize and search financial documents using Cohere's LLMs.

This DAG extracts and splits financial reporting data from the US 
[Securities and Exchanges Commision (SEC) EDGAR database](https://www.sec.gov/edgar) and generates 
vector embeddings with cohere embeddings for generative question answering.  The DAG also 
creates and vectorizes summarizations of the 10-Q document using Cohere Summarize.
"""
from __future__ import annotations

from airflow.decorators import dag, task
from airflow.exceptions import AirflowException
from airflow.models.param import Param
from airflow.providers.cohere.hooks.cohere import CohereHook

from bs4 import BeautifulSoup
from cohere.client import Client as CohereClient
import datetime
from langchain.schema import Document
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import logging
import pandas as pd
from pathlib import Path
import requests
import unicodedata
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

    def remove_html_tables(content:str):
        """
        Remove all "table" tags from html content leaving only text.

        :param content: html content
        :return: A string of extracted text from html without tables.
        """
        soup = BeautifulSoup(content, "lxml")

        for table in soup.find_all("table"):
            _ = table.replace_with(" ")
        soup.smooth()
        
        clean_text = unicodedata.normalize("NFKD", soup.text)

        return clean_text

    def get_html_content(doc_link: str) -> str:
        """
        A helper function to support pandas apply. Scrapes doc_link for html content.

        :param doc_link: Page url
        :return: Extracted plain text from html without any tables.
        """
        content = requests.get(doc_link, headers=edgar_headers)
        
        if content.ok:
            content_type = content.headers['Content-Type']
            if content_type == 'text/html':
                content = remove_html_tables(content.text)
            else:
                logger.warning(f"Unsupported content type ({content_type}) for doc {doc_link}.  Skipping.")
                content = None
        else:
            logger.warning(f"Unable to get content.  Skipping. Reason: {content.status_code} {content.reason}")
            content = None
        
        return content

    def get_10q_link(accn: str, cik_number: str) -> str:
        """
        Given an Accn number from SEC filings index, returns the URL of the 10-Q document.

        :param accn: account number for the filing
        :param cik_number: SEC Central Index Key for the company
        :return: Fully-qualified url pointing to a 10-Q filing document.
        """
        
        url_base = f"https://www.sec.gov/Archives/edgar/data/"

        link_base = f"{url_base}{cik_number}/{accn.replace('-','')}/"

        filing_summary = requests.get(f"{link_base}{accn}-index.html", headers=edgar_headers)

        link = None
        if filing_summary.ok:

            soup = BeautifulSoup(filing_summary.content, "lxml")

            for tr in soup.find("table", {"class": "tableFile"}).find_all("tr"):
                for td in tr.find_all('td'):
                    if td.text == "10-Q":
                        link = link_base + tr.find('a').text
        else:
            logger.warn(f"Error extracting accn index. Reason: {filing_summary.status_code} {filing_summary.reason}")

        return link

    def extract(ticker: str) -> pd.DataFrame:
        """
        This task pulls 10-Q statements from the [SEC Edgar database](https://www.sec.gov/edgar/searchedgar/companysearch)

        :param ticker: ticker symbol of company 
        :param cik_number: optionally cik_number instead of ticker symbol
        :return: A dataframe
        """

        logger.info(f"Extracting documents for ticker {ticker}.")

        company_list = requests.get(
            url="https://www.sec.gov/files/company_tickers.json", 
            headers=edgar_headers)

        if company_list.ok:
            company_list = list(company_list.json().values())
            cik_numbers = [item for item in company_list if item.get("ticker") == ticker.upper()]

            if len(cik_numbers) != 1:
                raise ValueError("Provided ticker symbol is not available.")
            else:
                cik_number = str(cik_numbers[0]['cik_str'])

        else:
            logger.error("Could not access ticker database.")
            logger.error(f"Reason: {company_list.status_code} {company_list.reason}")
            raise AirflowException("Could not access ticker database.")
        
        company_facts = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_number.zfill(10)}.json", 
            headers=edgar_headers
            )
            
        if company_facts.ok:
            forms_10q = []
            for fact in company_facts.json()['facts']['us-gaap'].values():
                for currency, units in fact['units'].items():
                    for unit in units:
                        if unit["form"] == "10-Q":
                            forms_10q.append(unit)

            forms_10q = pd.DataFrame(forms_10q)[["accn", "fy", "fp"]].drop_duplicates().to_dict('records')

        else:
            logger.error(f"Could not get company filing information for ticker: {ticker}, cik: {cik_number}.")
            logger.error(f"Reason: {company_facts.status_code} {company_facts.reason}")
            raise AirflowException(f"Could not get company filing information for ticker: {ticker}, cik: {cik_number}.")
            
        docs = []
        for form in forms_10q:
            link_10q = get_10q_link(accn=form.get("accn"), cik_number=cik_number)
            docs.append({
                "docLink": link_10q, 
                "tickerSymbol": ticker,
                "cikNumber": cik_number,
                "fiscalYear": form.get("fy"), 
                "fiscalPeriod": form.get("fp")
                })
            
        df = pd.DataFrame(docs)

        df["content"] = df.docLink.apply(lambda x: get_html_content(doc_link=x))
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df

    def split(df: pd.DataFrame) -> pd.DataFrame:
        """
        This task concatenates multiple dataframes from upstream dynamic tasks and splits the content 
        first with an html splitter and then with a text splitter.

        :param df: A dataframe from an upstream extract task
        :return: A dataframe 
        """

        headers_to_split_on = [
            ("h2", "h2"),
        ]

        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )

        df["doc_chunks"] = df["content"].apply(lambda x: html_splitter.split_text(text=x))
        df = df.explode("doc_chunks", ignore_index=True)
        df["content"] = df["doc_chunks"].apply(lambda x: x.page_content)

        df["doc_chunks"] = df["content"].apply(
            lambda x: text_splitter.split_documents([Document(page_content=x)])
            )
        df = df.explode("doc_chunks", ignore_index=True)
        df["content"] = df["doc_chunks"].apply(lambda x: x.page_content)

        df.drop(["doc_chunks"], inplace=True, axis=1)
        df.drop_duplicates(subset=["docLink", "content"], keep="first", inplace=True)
        df.reset_index(inplace=True, drop=True)

        return df

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
        
    def summarize_cohere(df: pd.DataFrame) -> pd.DataFrame:
        """
        This task uses cohere to recursively summarize extracted documents. First the individual
        chunks of the document are summarized.  Then the collection of chunk summaries are summarized.

        :param df: A Pandas dataframe from upstream split tasks
        :return: A Pandas dataframe with summaries for ingest to a vector DB.
        """

        cohere_client = CohereHook(COHERE_CONN_ID).get_conn

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

    edgar_docs = task(extract)(ticker=ticker)

    split_docs = task(split)(df=edgar_docs)

    embeddings_file = task(vectorize)(
        output_file_name="include/data/html/cohere_embeddings.parquet",
        content_column_name="content",
        df=split_docs)
    
    generate_summary = task(summarize_cohere)(df=split_docs)

    summaries_file = (
        task(vectorize, task_id="vectorize_summaries")(
            output_file_name="include/data/html/cohere_summary_embeddings.parquet",
            content_column_name="summary",
            df=generate_summary)
    )

FinSum_Cohere(ticker="")