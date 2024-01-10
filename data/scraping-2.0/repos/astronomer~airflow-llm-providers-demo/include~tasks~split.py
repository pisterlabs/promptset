from __future__ import annotations

from langchain.schema import Document
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import pandas as pd

def split_html(df: pd.DataFrame) -> pd.DataFrame:
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
