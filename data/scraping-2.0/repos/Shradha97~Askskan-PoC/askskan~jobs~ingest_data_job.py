"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader

from utils.api_keys.load_openai_key import load_openai_api_key

from app.configurations.development.settings import (
    SCHEMA_FILE,
    VECTORSTORE_FILE,
)


def file_paths():
    file0 = "data/sample/schema0.txt"
    file1 = "data/original/SchemaToShare.xlsx"

    return file1


def get_data():
    """
    FIX ME: This is not modular
    put this in an engines or other folder
    """

    file = file_paths()
    # Read the Excel file and skip the first row
    df = pd.read_excel(file, skiprows=1)

    # Drop the desired column
    column_to_drop = "Source"
    df = df.drop(column_to_drop, axis=1)

    # Convert DataFrame to CSV
    df.to_csv("/data/original/schema1.csv", index=False)
    file2 = "data/original/schema1.csv"

    return file2


def ingest_docs():
    """Get documents from the appropriate database."""
    file_path = SCHEMA_FILE  # get_data()

    loader = CSVLoader(file_path=file_path)
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(VECTORSTORE_FILE, "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    load_openai_api_key()
    ingest_docs()
