# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, missing-module-docstring, missing-function-docstring, W0621:redefined-outer-name,missing-class-docstring

import json
import os
import openai

from llama_index import (
    VectorStoreIndex,
    download_loader,
)
from llama_index.retrievers import BaseRetriever
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index import QueryBundle
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import BM25Retriever
from global_variable import QUERY, VECTOR_STORE_PATH_BASELINE

from utils import (
    create_storage_and_service_contexts,
    get_index_and_nodes_from_activeloop,
    get_pills_info,
    load_vector_store,
)


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
token = os.getenv("ACTIVELOOP_TOKEN")


PILLS_JSON_FILE_CLEANED = "pills_info_cleaned.json"


class ClassicRetrieverBM25(BaseRetriever):
    def __init__(self, bm25_retriever):
        # self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        # vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


if __name__ == "__main__":
    pills_info = get_pills_info()

    # IN CASE YOU WANT TO CREATE A NEW VECTOR STORE AND POPULATE IT WITH THE DOCUMENTS
    # (
    #     service_context,
    #     storage_context,
    #     nodes,
    #     llm,
    #     index,
    # ) = create_storage_and_service_contexts(
    #     vector_store_path=VECTOR_STORE_PATH_BASELINE,
    # )

    # IN CASE YOU WANT TO LOAD AN EXISTING VECTOR STORE
    _, nodes, service_context = get_index_and_nodes_from_activeloop(
        vector_store_path=VECTOR_STORE_PATH_BASELINE
    )

    # retireve the top 10 most similar nodes using bm25
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

    # Keep only the bm25 retriever
    hybrid_only_bm25_retriever = ClassicRetrieverBM25(bm25_retriever)

    # nodes retrieved by the bm25 retriever without the reranker
    nodes_bm25_response = hybrid_only_bm25_retriever.retrieve(QUERY)

    print(nodes_bm25_response)
    for el in nodes_bm25_response:
        print(f"{el.text}\n\n")

    reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

    # nodes retrieved by the bm25 retriever with the reranker
    reranked_nodes_bm25 = reranker.postprocess_nodes(
        nodes_bm25_response,
        query_bundle=QueryBundle(QUERY),
    )

    print("Initial retrieval: ", len(nodes_bm25_response), " nodes")
    for el in nodes_bm25_response:
        print(f"{el.score}\n\n")
    print("Re-ranked retrieval: ", len(reranked_nodes_bm25), " nodes")
    query_engine_bm25 = RetrieverQueryEngine.from_args(
        retriever=hybrid_only_bm25_retriever,
        node_postprocessors=[reranker],
        service_context=service_context,
    )

    response = query_engine_bm25.query(QUERY)
    print(response)
