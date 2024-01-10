import dataclasses
from pathlib import Path

from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import pprint
from typing import Iterable, Any, TypedDict

@dataclasses.dataclass
class _ClientConfig:
    host: str
    port: int
    prefer_grc: bool

@dataclasses.dataclass
class _IngestionSources:
    structured: list[str]
    unstructured: list[str]

@dataclasses.dataclass
class _IngestionSinks:
    collection_name: str

@dataclasses.dataclass
class _IngestionConfig:
    source_dir: str | Path
    sources: _IngestionSources
    sinks : _IngestionSinks

@dataclasses.dataclass
class _Config:
    client: _ClientConfig
    ingestion: _IngestionConfig

# TODO: 
# Make ingestion service that gets both embeddings and data from the source as documents for langchain
# Inject embeddings, docs and metadata as payload in the vectorstore
# Check fake embeddings in neural nine video. 
# Check llama index embeddings and integrations

def get_qdrant_client(config: _ClientConfig | None, in_memory: bool = False) -> QdrantClient:
    if not in_memory:
        if not config:
            raise ValueError("Qdrant client config not provided")
        return QdrantClient(host=config.host, port=config.port, prefer_grpc=config.prefer_grc)
    return QdrantClient(":memory:")

def get_qdrant_lc_client(client: QdrantClient, embeddings, config: _Config) -> Qdrant:
    x =  Qdrant(client, collection_name=config.ingestion.sinks.collection_name)
    x.

def generate_records(source_file: str, index_name: str) -> Iterable[dict[str, Any]]:
    df = pandas.read_excel(source_file)
    for record in df.to_dict(orient='records'):
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_source": record
        }

@timeit(fmt_msg="Data indexed successfully in {}s")
def main(config: IndexerConfig) -> list[tuple[bool, dict[str, Any]]]:
    return [
        result
        for result in helpers.streaming_bulk(
            client=get_elasticsearch(config.es_host, config.es_port), 
            actions=generate_records(config.source_file, config.index_name), 
            max_chunk_bytes=1000
        )
    ]

if __name__ == "__main__":
    print("Started indexing")
    config = IndexerConfig(
        es_host="localhost", 
        es_port=9200, 
        index_name="raw_index_001",
        source_file="~/Documents/dummy.xlsx",
    )
    result = main(config)
    print(f"Summary: {pprint.pformat(result, indent=2)}")
