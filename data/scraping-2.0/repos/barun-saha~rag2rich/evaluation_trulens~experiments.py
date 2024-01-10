import os
import pathlib
import random
import time
import numpy as np
import litellm

from langchain.llms import VertexAI
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer, SummaryIndex,
)
from llama_index.indices.postprocessor import CohereRerank
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.litellm import LiteLLM
from typing import List, Tuple, Iterable
from dotenv import load_dotenv

import helper.data
from helper import vertex_ai as vai_util


load_dotenv()
litellm.set_verbose = False


def get_lite_llm_provider() -> LiteLLM:
    """
    Get an instance of ChatVertexAI via LiteLLM.

    :return: The LLM provider
    """

    return LiteLLM(model_engine='chat-bison-32k', max_output_tokens=2048, temperature=0.0)


def load_questions(file_name: str) -> List[str]:
    """
    Read the questions from a file. Each line contains one questions.

    :param file_name: The input file name
    :return: The list of questions
    """

    with open(file_name, 'r') as in_file:
        return in_file.readlines()


def experiment_with_chunks(chunk_size_overlap=Iterable[List[Tuple[int]]]) -> None:
    """
    Run experiments with different chunk sizes and overlaps.

    :param chunk_size_overlap: A list of (chunk size, overlap) values
    """

    questions = load_questions('questions.txt')

    tru_llm = LiteLLM(model_engine='chat-bison-32k')
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    llm = VertexAI(
        model='text-bison',
        temperature=0,
        additional_kwargs=vai_util.VERTEX_AI_LLM_PARAMS
    )
    embeddings = vai_util.CustomVertexAIEmbeddings()

    n_configs = len(chunk_size_overlap)
    # https://cloud.google.com/vertex-ai/docs/quotas#generative-ai
    rate_limiter = vai_util.rate_limit(60)

    for idx, a_config in enumerate(chunk_size_overlap):
        chunk_size, chunk_overlap = a_config
        print(f'Config {idx + 1} of {n_configs}: {chunk_size=}, {chunk_overlap=}')

        service_context = ServiceContext.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            llm=llm,
            embed_model=embeddings
        )

        documents = SimpleDirectoryReader(
            input_files=['../data/TR-61850.pdf']
        ).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )

        app_id = f'RAG2Rich_LlamaIndex_App_{idx + 1}'
        query_engine = index.as_query_engine()
        tru_query_engine_recorder = TruLlama(
            query_engine,
            app_id=app_id,
            feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
        )

        with tru_query_engine_recorder as _:
            print('Running queries...')
            for a_question in questions:
                start_time = time.perf_counter()
                response = query_engine.query(a_question)
                end_time = time.perf_counter()
                print('Response:', response)
                print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
                next(rate_limiter)
                time.sleep(3)


def get_max_qpm(k: int) -> int:
    """
    Adaptively rate limit the LLm API calls for RAG triad based on the top-k value.
    Warning: it is not fool-proof!
    Also, it does not consider that LiteLLM may retry running the failed queries again.

    :param k: The top-k value for vector search
    :return: The maximum no. of queries to run per minute
    """

    # One each for the answer, answer relevance, groundedness; plus, some margin
    n_approx_calls = k + 3 + 2
    n_total_calls = max(1, int(60 / n_approx_calls))
    print(f'get_max_qpm: {k=}, {n_total_calls=}')

    return n_total_calls


def experiment_with_top_k(top_k=Iterable[List[int]], similarity_cutoff=Iterable[List[float]]) -> None:
    """
    Run experiments with different top-k values and similarity cutoff for vector search.

    :param top_k: A list of top-k values for vector search
    :param similarity_cutoff: Threshold for ignoring nodes
    """

    questions = load_questions('questions.txt')
    documents = SimpleDirectoryReader(
        input_files=['../data/TR-61850.pdf']
    ).load_data()

    tru_llm = get_lite_llm_provider()
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    n_configs = len(top_k) * len(similarity_cutoff)
    idx = 1

    service_context = helper.data.get_service_context(chunk_size=512, chunk_overlap=100)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    the_runs = []

    for k in top_k:
        rate_limiter = vai_util.rate_limit(max_per_minute=get_max_qpm(k))

        for c in similarity_cutoff:
            print(f'Config {idx} of {n_configs}: {k=}, {c=}')
            the_runs.append((k, c))

            # Configure retriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=k,
                service_context=service_context
            )

            # Configure response synthesizer
            response_synthesizer = get_response_synthesizer(service_context=service_context)

            # Assemble query engine
            if c is None:
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=c)],
                )
            else:
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=c)],
                )

            app_id = f'RAG2Rich_Exp_top_k={k}_cutoff={c}'
            tru_query_engine_recorder = TruLlama(
                query_engine,
                app_id=app_id,
                feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
            )

            with tru_query_engine_recorder as _:
                print('Running queries...')
                for a_question in questions:
                    start_time = time.perf_counter()
                    response = query_engine.query(a_question)
                    end_time = time.perf_counter()
                    print('Response:', response)
                    print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
                    next(rate_limiter)
                    # Additional buffer
                    time.sleep(1 + random.random())

        # Clear any previous timer
        time.sleep(60 + random.random())

    print('The runs are:')
    for idx, run in enumerate(the_runs):
        print(f'Index: {idx}:: {run}')


def experiment_with_chunks_and_top_k(
        chunk_size=Iterable[List[int]],
        chunk_overlap=Iterable[List[int]],
        top_k=Iterable[List[int]],
        similarity_cutoff=Iterable[List[float]]
) -> None:
    """
    Run experiments with different top-k values and similarity cutoff for vector search.

    :param chunk_size: The chunk size values
    :param chunk_overlap: The chunk overlap values
    :param top_k: A list of top-k values for vector search
    :param similarity_cutoff: Threshold for ignoring nodes
    """

    questions = load_questions('questions.txt')
    documents = SimpleDirectoryReader(
        input_files=['../data/TR-61850.pdf']
    ).load_data()

    tru_llm = get_lite_llm_provider()
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    n_configs = len(chunk_size) * len(chunk_overlap) * len(top_k) * len(similarity_cutoff)
    idx = 0
    the_runs = []

    for c_size in chunk_size:
        for c_overlap in chunk_overlap:
            for k in top_k:
                service_context = helper.data.get_service_context(chunk_size=c_size, chunk_overlap=c_overlap)
                index = VectorStoreIndex.from_documents(
                    documents,
                    service_context=service_context
                )
                rate_limiter = vai_util.rate_limit(max_per_minute=get_max_qpm(k))

                for c in similarity_cutoff:
                    print(f'Config {idx + 1} of {n_configs}: {c_size=}, {c_overlap=}, {k=}, {c=}')
                    the_runs.append((c_size, c_overlap, k, c))

                    # Configure retriever
                    retriever = VectorIndexRetriever(
                        index=index,
                        similarity_top_k=k,
                        service_context=service_context
                    )

                    # Configure response synthesizer
                    response_synthesizer = get_response_synthesizer(service_context=service_context)

                    # Assemble query engine
                    if c is None:
                        query_engine = RetrieverQueryEngine(
                            retriever=retriever,
                            response_synthesizer=response_synthesizer,
                            # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=c)],
                        )
                    else:
                        query_engine = RetrieverQueryEngine(
                            retriever=retriever,
                            response_synthesizer=response_synthesizer,
                            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=c)],
                        )

                    app_id = f'{idx:02d}. RAG2Rich_c_size={c_size}_c_overlap={c_overlap}_top_k={k}_cutoff={c}'
                    idx += 1

                    tru_query_engine_recorder = TruLlama(
                        query_engine,
                        app_id=app_id,
                        feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
                    )

                    with tru_query_engine_recorder as _:
                        print('Running queries...')
                        for a_question in questions:
                            start_time = time.perf_counter()
                            response = query_engine.query(a_question)
                            end_time = time.perf_counter()
                            print('Response:', response)
                            print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
                            next(rate_limiter)
                            # Additional buffer
                            time.sleep(1 + 1.2 * random.random())

                # Clear any previous timer
                time.sleep(60 + random.random())

    print('The runs are:')
    for idx, run in enumerate(the_runs):
        print(f'Index: {idx}:: {run}')


def experiment_with_reranker(
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        top_n: int
) -> None:
    """
    Run experiments using Cohere re-ranker, together with different top-k values and similarity cutoff for vector search.

    :param chunk_size: The chunk size value
    :param chunk_overlap: The chunk overlap value
    :param top_k: The top-k value for vector search
    :param top_n: Cohere top-n results
    """

    questions = load_questions('questions.txt')
    documents = SimpleDirectoryReader(
        input_files=['../data/TR-61850.pdf']
    ).load_data()

    tru_llm = get_lite_llm_provider()
    grounded = Groundedness(groundedness_provider=tru_llm)

    # Define a groundedness feedback function
    f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(tru_llm.relevance).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_qs_relevance = Feedback(tru_llm.qs_relevance).on_input().on(
        TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

    service_context = helper.data.get_service_context(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    rate_limiter = vai_util.rate_limit(max_per_minute=get_max_qpm(top_k))

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        service_context=service_context
    )

    cohere_rerank = CohereRerank(
        api_key=os.environ['COHERE_API_KEY'],
        top_n=top_n
    )

    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer(service_context=service_context)

    # Assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[cohere_rerank],
        # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=c)],
    )

    app_id = f'RAG2Rich_cohere_c_size={chunk_size}_c_overlap={chunk_overlap}_top_k={top_k}_top_n={top_n}'

    tru_query_engine_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
    )

    with tru_query_engine_recorder as _:
        print('Running queries...')
        for a_question in questions:
            start_time = time.perf_counter()
            response = query_engine.query(a_question)
            end_time = time.perf_counter()
            print('Response:', response)
            print(f'Computation time: {1000 * (end_time - start_time):.3f} ms')
            next(rate_limiter)
            # Additional buffer
            time.sleep(1 + random.random())


if __name__ == '__main__':
    tru = Tru()
    tru.start_dashboard(
        # force=True,  # Not supported on Windows
        _dev=pathlib.Path().cwd().parent.parent.resolve()
    )

    # If needed, you can reset the trulens_eval dashboard database
    # tru.reset_database()

    config_values = [
        (512, 50),
        (512, 100),
        (768, 50),
        (768, 100),
        (1024, 50),
        (1024, 100)
    ]

    # Uncomment to run the experiment
    # experiment_with_chunks(chunk_size_overlap=config_values)

    # Does not perform well
    # experiment_with_summary_index((512, 100))

    # The optimal top-k value is 6 without similarity cutoff
    # A positive value of similarity cutoff degraded the scores
    # experiment_with_top_k(top_k=[2, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])
    # experiment_with_top_k(top_k=[4, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])
    # experiment_with_top_k(top_k=[6, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])
    # experiment_with_top_k(top_k=[8, ], similarity_cutoff=[0.3, 0.4, 0.5, 0.6])

    # Optimal k = 2 (RAG2Rich_Exp_top_k=2_cutoff=None)
    # experiment_with_top_k(top_k=[2, 4, 6, ], similarity_cutoff=[None])

    # Connection got reset after running 13 out of 27 experiments
    # The optimal config is RAG2Rich_c_size=512_c_overlap=75_top_k=2_cutoff=None
    # chunk_size = 512, chunk_overlap = 75, top_k = 2
    # experiment_with_chunks_and_top_k(
    #     chunk_size=[512, 768, 1024, ],
    #     chunk_overlap=[50, 75, 100, ],
    #     top_k=[2, 4, 6, ],
    #     similarity_cutoff=[None]
    # )

    # experiment_with_chunks_and_top_k(
    #     chunk_size=[512, ],
    #     chunk_overlap=[75, ],
    #     top_k=[3, ],
    #     similarity_cutoff=[None]
    # )

    # experiment_with_reranker(
    #     chunk_size=512,
    #     chunk_overlap=75,
    #     top_k=3,
    #     top_n=2
    # )
    #
    # This offers optimal evaluation results
    # experiment_with_reranker(
    #     chunk_size=512,
    #     chunk_overlap=75,
    #     top_k=4,
    #     top_n=2
    # )
    #
    # experiment_with_reranker(
    #     chunk_size=512,
    #     chunk_overlap=75,
    #     top_k=4,
    #     top_n=3
    # )
