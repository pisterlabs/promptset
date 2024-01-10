from dagster import (
    AssetKey,
    Definitions,
    asset,
)
from dagster_airbyte import load_assets_from_airbyte_instance, AirbyteResource
from langchain.document_loaders import AirbyteJSONLoader
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import pickle

airbyte_instance = AirbyteResource(
    host="localhost",
    port="8000",
)

airbyte_assets = load_assets_from_airbyte_instance(
    airbyte_instance,
    key_prefix="airbyte_asset",
)

stream_name = "Account"

airbyte_loader = AirbyteJSONLoader(
    f"/tmp/airbyte_local/_airbyte_raw_{stream_name}.jsonl"
)


@asset(
    non_argument_deps={AssetKey(["airbyte_asset", stream_name])},
)
def raw_documents():
    return airbyte_loader.load()


@asset
def documents(raw_documents):
    return RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(
        raw_documents
    )


@asset
def vectorstore(documents):
    vectorstore_contents = FAISS.from_documents(documents, OpenAIEmbeddings())
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore_contents, f)


defs = Definitions(assets=[airbyte_assets, raw_documents, documents, vectorstore])
