import os

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import OpenAIEmbeddings

from kb_guardian.utils.paths import get_config

CONFIG = get_config()


def get_deployment_embedding() -> OpenAIEmbeddings:
    """
    Depending on the configuration, returns an embedding deployed on Azure or an embedding specified by the config file.

    When using AzureOpenAI, the DEPLOYMENT_EMBEDDING environment variable should be set.

    Returns:
        OpenAIEmbeddings: An embedding provided by OpenAI.
    """  # noqa: E501
    if CONFIG["azure_openai"]:
        deployment_embedding = os.getenv("DEPLOYMENT_EMBEDDING")
        return OpenAIEmbeddings(
            deployment=deployment_embedding,
            chunk_size=1,
        )
    else:
        return OpenAIEmbeddings(model=CONFIG["embedding_model"])


def get_deployment_llm() -> BaseChatModel:
    """
    Depending on the configuration, returns an LLM deployed on Azure or an LLM specified by the config file.

    When using AzureOpenAI, the DEPLOYMENT_LLM environment variable should be set.

    Returns:
        BaseChatModel: An LLM provided by OpenAI
    """  # noqa: E501
    if CONFIG["azure_openai"]:
        deployment_llm = os.getenv("DEPLOYMENT_LLM")
        return AzureChatOpenAI(
            deployment_name=deployment_llm, streaming=True, temperature=0
        )
    else:
        return ChatOpenAI(model_name=CONFIG["llm"], streaming=True, temperature=0)
