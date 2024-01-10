from langchain import OpenAI
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, QuestionAnswerPrompt, LLMPredictor, PromptHelper, ServiceContext, Document
import logging
import sys
import os
import argparse
from typing import Any, Dict, List, Optional, cast

from gpt_index import GPTListIndex, SimpleDirectoryReader
from traverse_code_files import  list_go_files_recursive
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index import GPTQdrantIndex
from gpt_index.vector_stores.qdrant import QdrantVectorStore
from gpt_index.vector_stores.simple import SimpleVectorStore
from gpt_index.utils import iter_batch
from qdrant_client.http import models as rest

from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQueryResult,
    VectorStoreQuery,
)

import qdrant_client
from read_key import read_key_from_file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Query Engine for KubeBlocks.")
    parser.add_argument("key_file", type=str, help="Key file for OpenAI_API_KEY.")
    return parser.parse_args()

def collection_exists(client: qdrant_client, collection_name: str) -> bool:
    """Check if a collection exists."""
    from grpc import RpcError
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        client.get_collection(collection_name)
    except (RpcError, UnexpectedResponse, ValueError):
        return False
    return True

def main():
    args = parse_arguments()
    key_file = args.key_file

    openai_api_key = read_key_from_file(key_file)
    # set env for OpenAI api key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # set log level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    client = qdrant_client.QdrantClient(
        url="http://localhost",
        port=6333,
        grpc_port=6334,
        prefer_grpc=False,
        https=False,
        timeout=300
    )

    collection_name = "kubeblocks_config"
    vector_index = GPTSimpleVectorIndex.load_from_disk('config.json')
    docs = vector_index.docstore.docs
    embedding_dict = vector_index.get_vector_store().get_embedding_dict()

    embedding_len = 0
    for doc_id, vector in embedding_dict.items():
        embedding_len = len(vector)
        print(f"embedding_len:{embedding_len}")
        break

    if len(docs.items()) > 0 and not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=embedding_len,
                distance=rest.Distance.COSINE,
            ),
        )

    qdrant_store = QdrantVectorStore(collection_name=collection_name, client=client)

    count = 0
    new_ids = []
    vectors = []
    payloads = []

    for doc_id, doc in docs.items():
        count += 1
        if count == 100:
            client.upsert(
                collection_name=collection_name,
                points=rest.Batch.construct(
                    ids=new_ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
            count = 0
        else:
            new_ids.append(doc_id)
            embedding = embedding_dict[doc_id]
            vectors.append(embedding)
            payloads.append(
                {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "extra_info": doc.extra_info,
                }
            )



if __name__ == "__main__":
    main()
