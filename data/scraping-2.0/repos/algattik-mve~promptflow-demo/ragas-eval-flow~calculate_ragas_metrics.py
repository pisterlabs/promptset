import os
import datasets

from promptflow import tool
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from dotenv import load_dotenv

from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLM

@tool
def calculate_ragas_metrics(
    question: str, answer: str, ground_truth: str, context: str
) -> dict:

    dataset = datasets.Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "ground_truths": [[ground_truth]],
            "contexts": [[context]],
        }
    )

    # load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

    openai_api_type = os.getenv("OPENAI_API_TYPE")
    openai_api_version = os.getenv("OPENAI_API_VERSION")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    chat_model = os.getenv("CHAT_MODEL")
    chat_deployment = os.getenv("CHAT_DEPLOYMENT")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")    

    if (
        openai_api_type is None
        or openai_api_version is None
        or openai_api_base is None
        or openai_api_key is None
        or chat_model is None
        or chat_deployment is None
        or embedding_model is None
        or embedding_deployment is None
    ):
        raise ValueError("One or more required environment variables are missing.")

    # list of metrics we're going to use
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        harmfulness,
    ]

    azure_openai_chat_model = AzureChatOpenAI(
        deployment_name=chat_deployment,
        model=chat_model,
        openai_api_base=openai_api_base,
        openai_api_type=openai_api_type,
    )

    # wrapper around azure_model
    ragas_chat_model = LangchainLLM(azure_openai_chat_model)

    for m in metrics:
        m.__setattr__("llm", ragas_chat_model)

    # init and change the embeddings
    # only for answer_relevancy
    azure_openai_embeddings_model = AzureOpenAIEmbeddings(
        deployment=embedding_deployment,
        model=embedding_model,
        openai_api_base=openai_api_base,
        openai_api_type=openai_api_type,
    )
    
    # embeddings can be used as it is
    answer_relevancy.embeddings = azure_openai_embeddings_model

    result = evaluate(
        dataset,
        metrics=metrics,
    )

    return result
