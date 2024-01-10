import logging
from dotenv import load_dotenv
from llama_index import VectorStoreIndex
import pandas as pd

from ragas.metrics import answer_relevancy
from ragas.llama_index import evaluate
from ragas.llms import LangchainLLM

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings


from app.llama_index.vector_store import setup_vector_store
from app.llama_index.llm import setup_service_context

from app.utils.env import get_env_variable
from app.eval.constants import (
    DATASET_JSON_PATH,
    EVAL_METRICS,
    EVAL_VECTOR_STORE_NAME,
    SERVICE_CONTEXT_VERSION,
)
from app.eval.dataset_generation import generate_ragas_qr_pairs


def setup_ragas_llm():
    load_dotenv()
    try:
        api_key = get_env_variable("OPENAI_API_KEY")
        api_version = get_env_variable("OPENAI_API_VERSION")
        deployment_name = get_env_variable("OPENAI_DEPLOYMENT_NAME")
    except EnvironmentError as e:
        raise e

    azure_model = AzureChatOpenAI(
        deployment_name=deployment_name,
        model=api_version,
        openai_api_key=api_key,
        openai_api_type="azure",
    )
    logging.info("Azure OpenAI model for Ragas successfully set up.")
    return LangchainLLM(azure_model)


def setup_ragas_embeddings():
    load_dotenv()
    try:
        deployment = get_env_variable("OPENAI_DEPLOYMENT_EMBEDDINGS")
        api_base = get_env_variable("OPENAI_API_BASE")
        api_key = get_env_variable("OPENAI_API_KEY")
        api_version = get_env_variable("OPENAI_API_VERSION")
    except EnvironmentError as e:
        raise e

    azure_embeddings = AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        model="text-embedding-ada-002",
        openai_api_type="azure",
        openai_api_base=api_base,
        openai_api_key=api_key,
        openai_api_version=api_version,
    )
    logging.info("Azure OpenAI Embeddings for Ragas successfully set up.")
    return azure_embeddings


def run_ragas_evaluation():
    eval_questions, eval_answers = generate_ragas_qr_pairs(DATASET_JSON_PATH)
    eval_embeddings = setup_ragas_embeddings()
    eval_llm = setup_ragas_llm()
    eval_vector_store = setup_vector_store(EVAL_VECTOR_STORE_NAME)
    eval_service_context = setup_service_context(SERVICE_CONTEXT_VERSION, azure=True)
    index = VectorStoreIndex.from_vector_store(
        vector_store=eval_vector_store, service_context=eval_service_context
    )
    query_engine = index.as_query_engine()
    logging.info("Ragas evaluation successfully set up.")

    metrics = EVAL_METRICS
    answer_relevancy.embeddings = eval_embeddings
    for m in metrics:
        m.__setattr__("llm", eval_llm)
        m.__setattr__("embeddings", eval_embeddings)
    logging.info("Ragas metrics successfully set up.")
    result = evaluate(query_engine, metrics, eval_questions, eval_answers)
    logging.info("Ragas evaluation successfully finished.")
    df = result.to_pandas()
    df.to_csv("app/eval/eval_data/ragas_eval.csv", index=False)
    logging.info("Ragas evaluation successfully saved to csv file.")
    eval = pd.read_csv("app/eval/eval_data/ragas_eval.csv", sep=",")
    logging.info("Ragas evaluation successfully finished.")
    return eval
