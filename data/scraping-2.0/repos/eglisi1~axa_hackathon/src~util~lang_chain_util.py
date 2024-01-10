import os
import openai

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings

from util.config import load_config

config = load_config("config/cfg.yaml")
interface = config["interface"]

openai.api_key = os.environ.get("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")


def create_llm() -> ChatOpenAI:
    from langchain.chat_models import ChatOpenAI

    return ChatOpenAI(
        temperature=config["situation_analysis"]["temperature"],
        model_name=config["situation_analysis"]["model_name"],
    )


def create_llm_chain(llm, prompt) -> LLMChain:
    return LLMChain(llm=llm, prompt=prompt, verbose=config["situation_analysis"]["verbose"])


def create_embedding() -> Embeddings:
    match interface:
        case "huggingface":
            from langchain.embeddings.huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=config["sentence_transformer"]["model_name"]
            )
        case "openai":
            from langchain.embeddings.openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model_name=config["sentence_transformer"]["model_name"]
            )
        case _:
            raise ValueError(f"Unknown interface '{interface}'")
