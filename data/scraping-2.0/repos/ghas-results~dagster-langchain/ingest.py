from dagster import (
    AssetIn,
    AssetSelection,
    Definitions,
    FreshnessPolicy,
    IOManager,
    asset,
    build_asset_reconciliation_sensor,
)
from dagster_airbyte import load_assets_from_airbyte_instance, AirbyteResource
from langchain.document_loaders import AirbyteJSONLoader
from langchain.text_splitter import TextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.document_loaders.base import BaseLoader, Document
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import pickle
from typing import List

airbyte_instance = AirbyteResource(
    host="localhost",
    port="8000",
)

airbyte_assets = load_assets_from_airbyte_instance(
    airbyte_instance,
    key_prefix="airbyte_asset",
)

stream_name="Account"

# Airbyte loader
airbyte_loader = AirbyteJSONLoader(
	f"/tmp/airbyte_local/_airbyte_raw_{stream_name}.jsonl"
)


@asset(
    name="raw_documents",
    ins={
        "airbyte_data": AssetIn(
            asset_key=["airbyte_asset", stream_name], input_manager_key="airbyte_io_manager"
        )
    },
    compute_kind="langchain",
)
def raw_documents(airbyte_data):
    """Load the raw document text from the source."""
    return airbyte_loader.load()


@asset(
    name="documents",
    ins={"raw_documents": AssetIn("raw_documents")},
    compute_kind="langchain",
)
def documents(raw_documents):
    """Split the documents into chunks that fit in the LLM context window."""
    return RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(raw_documents)


@asset(
    name="vectorstore",
    io_manager_key="vectorstore_io_manager",
    freshness_policy=FreshnessPolicy(maximum_lag_minutes=5, cron_schedule="@daily"),
    ins={"documents": AssetIn("documents")},
    compute_kind="vectorstore",
)
def vectorstore(documents):
    """Compute embeddings and create a vector store."""
    return FAISS.from_documents(documents, OpenAIEmbeddings())


class VectorstoreIOManager(IOManager):
    def load_input(self, context):
        raise NotImplementedError()

    def handle_output(self, context, obj):
        filename = "vectorstore.pkl"
        with open(filename, "wb") as f:
            pickle.dump(vectorstore, f)
        context.add_output_metadata({"filename": filename})


class AirbyteIOManager(IOManager):
    def load_input(self, context):
        return context.upstream_output.metadata

    def handle_output(self, context, obj):
        raise NotImplementedError()

defs = Definitions(
    assets=[airbyte_assets, raw_documents, documents, vectorstore],
    resources={
        "vectorstore_io_manager": VectorstoreIOManager(),
        "airbyte_io_manager": AirbyteIOManager(),
    },
    sensors=[
        build_asset_reconciliation_sensor(
            AssetSelection.all(),
            name="reconciliation_sensor",
        )
    ],
)
