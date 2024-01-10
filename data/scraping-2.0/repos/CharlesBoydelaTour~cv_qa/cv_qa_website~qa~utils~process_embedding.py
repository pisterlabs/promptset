import os
import pandas as pd
from typing import List
from .bot_utils import (
    Configuration,
    azure_search_init,
    query_with_azure_search,
    load_embedding,
)  # need to add utils. when running from main
from langchain.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch
from langchain.docstore.document import Document
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def read_csv_file(config: Configuration, delimiter: str = ";") -> pd.DataFrame:
    """
    Read a CSV file and return it as a DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.
    - delimiter (str): The delimiter used in the CSV file. Default is ";".

    Returns:
    - pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    config = config.config
    file_path = config["DATA"]["CONTEXT"]
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, header=0)
        print(data.head(10))
        return data
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"The file at path {file_path} does not exist."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The file at path {file_path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Error occurred while parsing the file at path {file_path}."
        ) from exc
    except Exception as exc:
        raise exc


def df_to_documents(to_store_df: pd.DataFrame) -> List[Document]:
    """
    Convert a DataFrame to a list of documents.
    Args:
        to_store_df (pd.DataFrame): The DataFrame to convert.
    Returns:
        List[Document]: A list of documents.
    """
    # Verify that "Title" column exists
    if "Title" not in to_store_df.columns:
        raise ValueError("The input DataFrame must contain a 'Title' column.")
    if "Description" not in to_store_df.columns:
        raise ValueError("The input DataFrame must contain a 'Description' column.")
    # replace "Description" with "page_content"
    to_store_df.rename(columns={"Description": "page_content"}, inplace=True)

    # Convert the DataFrame to a list of dictionaries
    to_store_dict = to_store_df.to_dict("records")
    print(to_store_dict)
    # Convert the list of dictionaries to a list of Documents
    to_store_docs = []
    for doc in to_store_dict:
        if "page_content" not in doc:
            doc["page_content"] = ""
        metadata = {"title": doc.get("Title", "")}
        doc["metadata"] = metadata
        to_store_docs.append(Document(**doc))

    return to_store_docs


def store_in_db(
    config: Configuration, to_store_df: pd.DataFrame
) -> AzureCosmosDBVectorSearch:
    """
    Store the documents in the vector store.
    Args:
        config (Configuration): A configuration object containing API credentials and settings.
        to_store_df (pd.DataFrame): The DataFrame to convert.
    Returns:
        AzureCosmosDBVectorSearch: A vector store object.
    """
    CONNECTION_STRING = os.getenv("MANGO_CONNECTION")
    DB_NAME = config.config["MONGODB"]["DB_NAME"]
    COLLECTION_NAME = config.config["MONGODB"]["COLLECTION_NAME"]
    INDEX_NAME = config.config["MONGODB"]["INDEX_NAME"]
    client: MongoClient = MongoClient(CONNECTION_STRING)
    collection = client[DB_NAME][COLLECTION_NAME]
    # if not empty, delete all documents in collection
    if collection.count_documents({}) > 0:
        collection.delete_many({})

    embedding_model = load_embedding(config)
    documents = df_to_documents(to_store_df)

    # Store the documents in the vector store
    vectorstore = AzureCosmosDBVectorSearch.from_documents(
        documents, embedding_model, collection=collection, index_name=INDEX_NAME
    )

    return vectorstore


def main_process(config):
    text_df = read_csv_file(config=config)
    vectorstore = store_in_db(config=config, to_store_df=text_df)
    # query = "Where did Charles studied?"
    # azure_search = azure_search_init(
    #    config=config,
    #    embedding_model=load_embedding(config=config),
    # )
    # results = query_with_azure_search(azure_search=azure_search, query=query)
    # print(results)
    return None


if __name__ == "__main__":
    text_df = read_csv_file(config=Configuration("config.json"))
    vectorstore = store_in_db(config=Configuration("config.json"), to_store_df=text_df)
    query = "Where did Charles studied?"
    azure_search = azure_search_init(
        config=Configuration("config.json"),
        embedding_model=load_embedding(config=Configuration("config.json")),
    )
    results = query_with_azure_search(azure_search=azure_search, query=query)
