# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, missing-module-docstring, missing-function-docstring, W0621:redefined-outer-name,missing-class-docstring
import os
import openai
from llama_index import (
    QueryBundle,
)
from llama_index.retrievers import BM25Retriever


from llama_index.retrievers import BaseRetriever
from llama_index.postprocessor import SentenceTransformerRerank
from llamaindex_bm25_baseline import ClassicRetrieverBM25
from llama_index.query_engine import RetrieverQueryEngine
from global_variable import (
    QUERY,
    VECTOR_STORE_PATH_COMPLETE_SEQUENTIALLY,
)

from utils import (
    create_storage_and_service_contexts,
    get_index_and_nodes_from_activeloop,
    get_pills_info,
    keep_best_k_unique_nodes,
    load_vector_store,
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
token = os.getenv("ACTIVELOOP_TOKEN")


class HybridRetrieverOnlyVector(BaseRetriever):
    """
    Retrieves bm25 nodes and vector nodes individually
    """

    def __init__(self, vector_retriever):
        self.vector_retriever = vector_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        all_nodes = []
        node_ids = set()
        for n in vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


class BM25CompleteSequentially:
    """
    The retriever are the same as the ones used in the baseline.
    Here the difference is that we retrieve the nodes sequentially and not in parallel and rerank them.
    """

    def __init__(self):
        self.vector_retriever = None
        self.bm25_retriever = None
        self.hybrid_only_bm25_retriever = None
        self.hybrid_only_vector_retriever = None
        self.nodes_bm25_response = None
        self.nodes_vector_response = None
        self.reranker = None
        super().__init__()

    def create_retrievers(self):
        index, nodes, _ = get_index_and_nodes_from_activeloop(
            vector_store_path=VECTOR_STORE_PATH_COMPLETE_SEQUENTIALLY
        )
        self.vector_retriever = index.as_retriever(similarity_top_k=2)
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=10
        )
        return self.vector_retriever, self.bm25_retriever

    def get_retrievers_and_rerunk(self, rerenk_model="BAAI/bge-reranker-base"):
        # query_gen_prompt="...",  # we could override the query generation prompt here
        self.hybrid_only_bm25_retriever = ClassicRetrieverBM25(self.bm25_retriever)
        self.hybrid_only_vector_retriever = HybridRetrieverOnlyVector(
            self.vector_retriever
        )
        self.reranker = SentenceTransformerRerank(top_n=4, model=rerenk_model)

        return (
            self.hybrid_only_bm25_retriever,
            self.hybrid_only_vector_retriever,
            self.reranker,
        )

    def get_rerunked_nodes(self, query):
        self.nodes_bm25_response = self.hybrid_only_bm25_retriever.retrieve(query)
        self.nodes_vector_response = self.hybrid_only_vector_retriever.retrieve(query)

        reranked_nodes_bm25 = self.reranker.postprocess_nodes(
            self.nodes_bm25_response,
            query_bundle=QueryBundle(QUERY),
        )
        print("Reranked Nodes BM25\n\n")
        for el in reranked_nodes_bm25:
            print(f"{el.score}\n")

        reranked_nodes_vector = self.reranker.postprocess_nodes(
            self.nodes_vector_response,
            query_bundle=QueryBundle(QUERY),
        )
        print("Reranked Nodes Vector\n\n")
        for el in reranked_nodes_vector:
            print(f"{el.score}\n")
            unique_nodes = keep_best_k_unique_nodes(
                reranked_nodes_bm25, reranked_nodes_vector
            )
            print("Unique Nodes\n\n")
            for el in unique_nodes:
                print(f"{el.id} : {el.score}\n")
        return unique_nodes


if __name__ == "__main__":
    pills_info = get_pills_info()

    vector_store = load_vector_store(VECTOR_STORE_PATH_COMPLETE_SEQUENTIALLY)

    # IN CASE YOU WANT TO CREATE A NEW VECTOR STORE AND POPULATE IT WITH THE DOCUMENTS
    # (
    #     service_context,
    #     storage_context,
    #     nodes,
    #     llm,
    #     index,
    # ) = create_storage_and_service_contexts(vector_store_path=VECTOR_STORE_PATH_COMPLETE_SEQUENTIALLY)

    # -------------old method----------------
    # IN CASE YOU WANT TO LOAD AN EXISTING VECTOR STORE
    # index, nodes, service_context = get_index_and_nodes_from_activeloop(
    #     vector_store_path=VECTOR_STORE_PATH_HYBRID
    # )

    # hybrid_only_bm25_retriever = HybridRetriever_ONLY_BM25(bm25_retriever)
    # hybrid_only_vector_retriever = HybridRetrieverOnlyVector(vector_retriever)

    # nodes_bm25_response = hybrid_only_bm25_retriever.retrieve(QUERY)
    # nodes_vector_response = hybrid_only_vector_retriever.retrieve(QUERY)

    # reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

    # SEQUENTIALLY

    # reranked_nodes_bm25 = reranker.postprocess_nodes(
    #     nodes_bm25_response,
    #     query_bundle=QueryBundle(QUERY),
    # )
    # print("Reranked Nodes BM25\n\n")
    # for el in reranked_nodes_bm25:
    #     print(f"{el.score}\n")

    # reranked_nodes_vector = reranker.postprocess_nodes(
    #     nodes_vector_response,
    #     query_bundle=QueryBundle(QUERY),
    # )
    # print("Reranked Nodes Vector\n\n")
    # for el in reranked_nodes_vector:
    #     print(f"{el.score}\n")

    # unique_nodes = keep_best_k_unique_nodes(reranked_nodes_bm25, reranked_nodes_vector)
    # print("Unique Nodes\n\n")
    # for el in unique_nodes:
    #     print(f"{el.id} : {el.score}\n")

    # -------------NEW method----------------

    bm25_complete_sequentially = BM25CompleteSequentially()
    vector_retriever, bm25_retriever = bm25_complete_sequentially.create_retrievers()
    (
        hybrid_only_bm25_retriever,
        hybrid_only_vector_retriever,
        reranker,
    ) = bm25_complete_sequentially.get_retrievers_and_rerunk()
    unique_nodes = bm25_complete_sequentially.get_rerunked_nodes(QUERY)

    # ----- NOT USEFUL FOR THIS USE CASE
    # query_engine_bm25 = RetrieverQueryEngine.from_args(
    #     retriever=hybrid_only_bm25_retriever,
    #     node_postprocessors=[reranker],
    #     service_context=service_context,
    # )

    # query_engine_vector = RetrieverQueryEngine.from_args(
    #     retriever=hybrid_only_vector_retriever,
    #     node_postprocessors=[reranker],
    #     service_context=service_context,
    # )

    # response = query_engine_bm25.query(QUERY)
    # print(response)

    # response = query_engine_vector.query(QUERY)
    # print(response)
