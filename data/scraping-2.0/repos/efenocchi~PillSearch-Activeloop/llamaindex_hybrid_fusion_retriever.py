# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, missing-module-docstring, missing-function-docstring, W0621:redefined-outer-name,missing-class-docstring

import os
from llama_index.retrievers import QueryFusionRetriever
from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
import openai
from global_variable import QUERY, VECTOR_STORE_PATH_BASELINE, VECTOR_STORE_PATH_HYBRID


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


class HybridFusionRetriever:
    def __init__(self):
        self.vector_retriever = None
        self.bm25_retriever = None
        super().__init__()

    # TODO Modify VECTOR_STORE_PATH_BASELINE
    def create_retrievers(self):
        index, nodes, _ = get_index_and_nodes_from_activeloop(
            vector_store_path=VECTOR_STORE_PATH_BASELINE
        )
        self.vector_retriever = index.as_retriever(similarity_top_k=2)
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=10
        )
        return self.vector_retriever, self.bm25_retriever

    def get_retriever(self):
        retriever = QueryFusionRetriever(
            [self.vector_retriever, self.bm25_retriever],
            similarity_top_k=2,
            num_queries=1,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            # query_gen_prompt="...",  # we could override the query generation prompt here
        )
        return retriever

    def get_nodes_with_scores(self):
        return self.get_retriever().retrieve(QUERY)


if __name__ == "__main__":
    pills_info = get_pills_info()

    vector_store = load_vector_store(VECTOR_STORE_PATH_HYBRID)

    # IN CASE YOU WANT TO CREATE A NEW VECTOR STORE AND POPULATE IT WITH THE DOCUMENTS
    # (
    #     service_context,
    #     storage_context,
    #     nodes,
    #     llm,
    #     index,
    # ) = create_storage_and_service_contexts(vector_store_path=VECTOR_STORE_PATH_HYBRID)

    # -------------old method----------------
    # IN CASE YOU WANT TO LOAD AN EXISTING VECTOR STORE
    # index, nodes, service_context = get_index_and_nodes_from_activeloop(
    #     vector_store_path=VECTOR_STORE_PATH_HYBRID
    # )

    # vector_retriever = index.as_retriever(similarity_top_k=2)

    # bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

    # retriever = QueryFusionRetriever(
    #     [vector_retriever, bm25_retriever],
    #     similarity_top_k=2,
    #     num_queries=1,  # set this to 1 to disable query generation
    #     mode="reciprocal_rerank",
    #     use_async=True,
    #     verbose=True,
    #     # query_gen_prompt="...",  # we could override the query generation prompt here
    # )
    # nodes_with_scores = retriever.retrieve(QUERY)

    # -------------NEW method----------------
    hybrid_retriever = HybridFusionRetriever()
    vector_retriever, bm25_retriever = hybrid_retriever.create_retrievers()
    retriever = hybrid_retriever.get_retriever()
    nodes_with_scores = hybrid_retriever.get_nodes_with_scores()

    print("Hybrid Fusion Nodes\n\n")
    for el in nodes_with_scores:
        print(f"{el.score}\n")
    query_engine = RetrieverQueryEngine.from_args(retriever)

    response = query_engine.query(QUERY)
