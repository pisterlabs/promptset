# Starter Code if needed
import logging
import sys
import time
from typing import List

import pandas as pd

# Setting up logs, but not really using at the moment
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from IPython.display import Markdown, display
from langchain.chat_models import ChatOpenAI

from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.evaluation import ResponseEvaluator

# set maximum input size
max_input_size = 4096
# set number of output tokens
num_outputs = 256
# set maximum chunk overlap
max_chunk_overlap = 20
# set chunk size limit
chunk_size = 600


def construct_index(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()

    # LLM Predictor (gpt-3.5-turbo) + service context
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            max_tokens=num_outputs,
            streaming=True,
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size=600
    )

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    # save index to disk
    index.storage_context.persist(persist_dir="/content/")

    return index


# previously named display_eval_df
def create_eval_df_2(
    query: str,
    response: Response,
    source_eval_result: List[str],
    response_eval_result: str,
):
    sources = [s.node.get_text() for s in response.source_nodes]
    eval_df = pd.DataFrame(
        {
            "Source": sources,
            "Source Eval Result": source_eval_result,
        },
    )
    eval_df["Query"] = query
    eval_df["Response"] = str(response)
    eval_df["Response Eval Result"] = response_eval_result
    eval_df = eval_df[
        ["Query", "Response", "Response Eval Result", "Source", "Source Eval Result"]
    ]

    return eval_df


def multi_prompt_ask_exam_assist(prompts: List[str]):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="/content/")
    # load index
    index = load_index_from_storage(storage_context)
    # LLM Predictor (gpt-3.5-turbo) + service context
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            max_tokens=num_outputs,
            streaming=True,
        )
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size=600
    )
    evaluator = ResponseEvaluator(service_context=service_context)
    experiments_df = pd.DataFrame()

    for x in prompts:
        user_input = x
        query_engine = index.as_query_engine(
            service_context=service_context,
            similarity_top_k=3,
            streaming=True,
        )

        response = query_engine.query(user_input)
        response.print_response_stream()
        source_eval_result = evaluator.evaluate_source_nodes(response)
        response_eval_result = evaluator.evaluate(response)

        eval_df = create_eval_df_2(
            user_input, response, source_eval_result, response_eval_result
        )

        experiments_df = experiments_df.append(eval_df)

        # Wait 5 seconds before starting the loop over, this can help avoid api overload errors
        time.sleep(10)

    return experiments_df
