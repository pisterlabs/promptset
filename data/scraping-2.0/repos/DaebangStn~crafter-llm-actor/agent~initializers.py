import os
from logging import Logger
from typing import List

from langchain.base_language import BaseLanguageModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever
from langchain.vectorstores import Chroma


def initialize_llm(configs: dict, secrets: dict, logger: Logger) -> BaseLanguageModel:
    llm_config = configs["defaults"]["llm"]

    if configs[llm_config]["provider"] == "huggingface":
        from langchain import HuggingFaceHub

        os.environ["HUGGINGFACEHUB_API_TOKEN"] = secrets["api_key"]["huggingface"]
        llm = HuggingFaceHub(repo_id=configs[llm_config]["model_id"], model_kwargs={"max_length": 64, "temperature": 0})
        logger.info("HuggingFaceHub initialized")
    elif configs[llm_config]["provider"] == "openai":
        os.environ["OPENAI_API_KEY"] = secrets["api_key"]["openai"]
        model_id = configs[llm_config]["model_id"]

        if model_id == "gpt-3.5-turbo":
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(model_name=model_id, temperature=0)
        else:
            from langchain import OpenAI
            llm = OpenAI(model_name=model_id, temperature=0)
        token_usage_limit = configs[llm_config]["token_usage_limit"]
        logger.info("OpenAI (%s) initialized. token_usage_limit: %s",  model_id, token_usage_limit)
    elif configs[llm_config]["provider"] == "mipal10":
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["OPENAI_API_BASE"] = secrets["urls"]["mipal10"]

        model_id = configs[llm_config]["model_id"]

        if model_id == "gpt-3.5-turbo":
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI(model_name=model_id, temperature=0)
        else:
            from langchain import OpenAI
            llm = OpenAI(model_name=model_id, temperature=0)

        token_usage_limit = configs[llm_config]["token_usage_limit"]
        logger.info("MIPAL LLM (%s) initialized. token_usage_limit: %s",  model_id, token_usage_limit)
    else:
        raise NotImplementedError

    assert llm is not None, "LLM not initialized"

    return llm


def initialize_retrievers(configs: dict, secrets: dict, logger: Logger) -> List[BaseRetriever]:
    llm_config = configs["defaults"]["llm"]
    provider = configs[llm_config]["provider"]
    assert provider == "openai" or provider == "mipal10", f"{provider} retriever is not supported"

    retrieversList = []
    collections = configs["defaults"]["chroma_retriever_name"]
    collections = collections.split(",")
    collections = [c.strip() for c in collections]

    os.environ["OPENAI_API_KEY"] = secrets["api_key"]["openai"]

    embedding = OpenAIEmbeddings()

    chroma_dir_path = configs["defaults"]["chroma_dir_path"]
    assert os.path.isdir(chroma_dir_path), "chroma_dir_path not found"

    for collection in collections:
        retriever = Chroma(
            embedding_function=embedding,
            persist_directory=chroma_dir_path,
            collection_name=collection,
        ).as_retriever()
        retrieversList.append(retriever)

    logger.debug("Loaded Chroma retrievers: %s", collections)

    return retrieversList
