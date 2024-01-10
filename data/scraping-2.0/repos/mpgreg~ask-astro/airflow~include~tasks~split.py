from __future__ import annotations

import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
)


def split_markdown(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    This task concatenates multiple dataframes from upstream dynamic tasks and splits the documents before importing
    to a vector database.

    :param: dfs: A list of dataframes from downstream dynamic tasks
    :return: A dataframe 
    """

    df = pd.concat(dfs, axis=0, ignore_index=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])

    df["doc_chunks"] = df["content"].apply(lambda x: splitter.split_documents([Document(page_content=x)]))
    df = df.explode("doc_chunks", ignore_index=True)
    df["content"] = df["doc_chunks"].apply(lambda x: x.page_content)
    df.drop(["doc_chunks"], inplace=True, axis=1)
    df.drop_duplicates(subset=["docLink", "content"], keep="first", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def split_python(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    This task concatenates multiple dataframes from upstream dynamic tasks and splits python code before importing
    to a vector database.

    param dfs: A list of dataframes from downstream dynamic tasks
    :return: A dataframe
    """

    df = pd.concat(dfs, axis=0, ignore_index=True)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        # chunk_size=50,
        chunk_overlap=0,
    )

    df["doc_chunks"] = df["content"].apply(lambda x: splitter.split_documents([Document(page_content=x)]))
    df = df.explode("doc_chunks", ignore_index=True)
    df["content"] = df["doc_chunks"].apply(lambda x: x.page_content)
    df.drop(["doc_chunks"], inplace=True, axis=1)
    df.drop_duplicates(subset=["docLink", "content"], keep="first", inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def split_html(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    This task concatenates multiple dataframes from upstream dynamic tasks and splits the content first with an 
    html splitter and then with a text splitter.

    :param dfs: A list of dataframes from downstream dynamic tasks
    :return: A dataframe 
    """

    headers_to_split_on = [
        ("h2", "h2"),
    ]

    df = pd.concat(dfs, axis=0, ignore_index=True)

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
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
