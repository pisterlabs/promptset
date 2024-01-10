# Langchain wraps the Redis client and provides a few convenience methods for working with documents.
# It can split documents into chunks, embed them, and store them in Redis.

import os
import argparse

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader
import logging

import redis
from redis.client import Redis as RedisType
from langchain.vectorstores.redis import Redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from typing import List


text_field = "otext"
primary_field = "id"
vector_field = "embedding"

# required modules
REDIS_REQUIRED_MODULES = [
    {"name": "search", "ver": 20400},
]


def _redis_prefix(index_name: str) -> str:
    """Redis key prefix for a given index."""
    return f"doc:{index_name}"


def _check_redis_module_exist(client: RedisType, modules: List[dict]) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {
        module[b"name"].decode("utf-8"): module for module in installed_modules
    }
    for module in modules:
        if module["name"] not in installed_modules or int(
            installed_modules[module["name"]][b"ver"]
        ) < int(module["ver"]):
            error_message = (
                "You must add the RediSearch (>= 2.4) module from Redis Stack. "
                "Please refer to Redis Stack docs: https://redis.io/docs/stack/"
            )
            logging.error(error_message)
            raise ValueError(error_message)


def load_documents(file_path, encoding='utf8', file_type='text'):
    if file_type == 'markdown':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def index_documents(redis_vector, docs):
    # Index the documents using the provided Redis instance
    redis_vector.add_documents(docs)


def create_redis_index(embeddings, index_name, username, password, host, port,
                       content_key: str = "content",
                       metadata_key: str = "metadata",
                       vector_key: str = "content_vector"):

    redis_url = "redis://{}:{}@{}:{}".format(username, password, host, port)

    try:
        client = redis.from_url(url=redis_url)
        # check if redis has redisearch module installed
        _check_redis_module_exist(client, REDIS_REQUIRED_MODULES)
    except ValueError as e:
        raise ValueError(f"Redis failed to connect: {e}")

    prefix = _redis_prefix(index_name)
    dim = 1536
    distance_metric = (
        "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)
    )
    schema = (
        TextField(name=content_key),
        TextField(name=metadata_key),
        VectorField(
            vector_key,
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dim,
                "DISTANCE_METRIC": distance_metric,
            },
        ),
    )
    # Create Redis Index
    client.ft(index_name).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
    )

    # Create the VectorStore
    redis_vector = Redis(
        redis_url,
        index_name,
        embeddings.embed_query,
        content_key=content_key,
        metadata_key=metadata_key,
        vector_key=vector_key,
    )

    return redis_vector


def main(input_dir, encoding, chunk_size, chunk_overlap, username, password, host, port, file_type, index_name):
    embeddings = OpenAIEmbeddings()
    redis_vector = create_redis_index(
        embeddings, index_name, username, password, host, port)
    # Iterate through all the files in the input directory and process each one
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            print(f"Processing {file_path}...")
            documents = load_documents(file_path, encoding, file_type)
            docs = split_documents(documents, chunk_size, chunk_overlap)
            index_documents(redis_vector, docs)
            print(f"Indexed {len(docs)} chunks from {file_path}.")


# python index_documents.py --input_dir /path/to/your/documents --file_type markdown --index_name index_name
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index documents for Question Answering over Documents application.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing documents to be indexed.')
    parser.add_argument('--encoding', type=str, default='utf8',
                        help='Encoding of the input documents.')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of the chunks to split documents into.')
    parser.add_argument('--chunk_overlap', type=int, default=0,
                        help='Number of overlapping characters between consecutive chunks.')
    parser.add_argument('--username', type=str,
                        help='username for the Redis server.')
    parser.add_argument('--password', type=str,
                        help='password for the Redis server.')
    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help='Host address for the Redis server.')
    parser.add_argument('--port', type=str, default="6379",
                        help='Port for the Redis server.')
    parser.add_argument('--file_type', type=str, default="text", choices=[
                        "text", "markdown"], help='Type of the input files (text or markdown).')
    parser.add_argument('--index_name', type=str, required=True,
                        help='Name of the Index to index the documents into.')

    args = parser.parse_args()

    main(args.input_dir, args.encoding, args.chunk_size, args.chunk_overlap,
         args.username, args.password, args.host, args.port, args.file_type, args.index_name)
