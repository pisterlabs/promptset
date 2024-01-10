from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.llms.fake import FakeListLLM

from langchain.embeddings import OpenAIEmbeddings, DeterministicFakeEmbedding
from langchain.chat_models import ChatOpenAI, ChatAnyscale

import tiktoken
import builtins
import logging

from abc import abstractmethod
from pydantic.v1 import BaseModel, Field, Extra, validator
from typing import Any, Optional

from enum import Enum

logger = logging.getLogger(__name__)

####################
# LLM Models
####################
LLAMA_2_MODELS = [
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
]

OPENAI_LLM_MODELS = ["gpt-3.5-turbo", "gpt-4"]

LLM_MODELS = [*OPENAI_LLM_MODELS, *LLAMA_2_MODELS]

# Anyscale configs
ANYSCALE_LLM_PREFIX = "meta-llama/"
ANYSCALE_API_URL = "https://api.endpoints.anyscale.com/v1"

####################
# Embedding Models
####################
OPENAI_EMB_MODELS = ["text-embedding-ada-002"]

EMB_MODELS = [*OPENAI_EMB_MODELS]


####################
# Enumerations
####################
class CVGradeAnswerPrompt(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    NONE = "none"


class CVGradeRetrieverPrompt(Enum):
    DEFAULT = "default"
    NONE = "none"


class CVRetrieverSearchType(Enum):
    BY_SIMILARITY = "similarity"
    MAXIMAL_MARGINAL_RELEVANCE = "mmr"


class CVSimilarityMethod(Enum):
    COSINE = "cosine"
    L2_NORM = "l2"
    INNER_PRODUCT = "ip"


####################
# Test LLM and embedding model for mocks
####################
class TestDummyLLM(FakeListLLM):
    """Langchains FakeListLLM with model name included."""

    model_name: str = "TestDummyLLM"

    def __init__(self):
        super().__init__(responses=["foo_response"])

    def dict(self, *args, **kwargs):
        output = super().dict(*args, **kwargs)
        output["model_name"] = self.model_name
        return output


class TestDummyEmbedding(DeterministicFakeEmbedding):
    """Langchains DeterministicFakeEmbedding with model name included."""

    model: str = "TestDummyEmbedding"

    def __init__(self):
        super().__init__(size=2)


class BaseConfigurations(BaseModel):
    """Base class for configuration objects."""

    chunk_size: int = Field(ge=0)
    chunk_overlap: int = Field(ge=0)
    length_function_name: str
    length_function: Any

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid

    @validator("length_function", pre=False, always=True)
    def populate_length_function(cls, v: callable, values: dict[str, str]):
        return cls.set_length_function(values["length_function_name"])

    @staticmethod
    def get_language_model(model_name: str, api_keys: dict) -> BaseLanguageModel:
        if model_name in OPENAI_LLM_MODELS:
            return ChatOpenAI(
                openai_api_key=api_keys["OPENAI_API_KEY"],
                model_name=model_name,
                temperature=0.0,
            )

        elif model_name in LLAMA_2_MODELS:
            return ChatAnyscale(
                anyscale_api_key=api_keys["ANYSCALE_API_KEY"],
                model_name=f"{ANYSCALE_LLM_PREFIX}{model_name}",
                anyscale_api_base=ANYSCALE_API_URL,
                temperature=0.0,
            )
        # only for testing purposes
        elif model_name == "TestDummyLLM":
            return TestDummyLLM()

        raise NotImplementedError(f"LLM model '{model_name}' not supported.")

    @staticmethod
    def get_embedding_model(model_name: str, api_keys: dict) -> Embeddings:
        if model_name in OPENAI_EMB_MODELS:
            return OpenAIEmbeddings(
                openai_api_key=api_keys["OPENAI_API_KEY"], model=model_name
            )
        elif model_name == "TestDummyEmbedding":
            return TestDummyEmbedding()

        raise NotImplementedError(f"Embedding model '{model_name}' not supported.")

    @classmethod
    def set_length_function(cls, length_function_name: str) -> callable:
        # Extract the function name from the string
        func = length_function_name.strip("<>").split(" ")[-1]

        # Check if the function name exists in Python's built-ins
        if hasattr(builtins, func):
            return getattr(builtins, func)

        else:
            try:
                encoding = tiktoken.encoding_for_model(length_function_name)
                return lambda x: len(encoding.encode(x))
            except Exception as ex:
                logger.error(f"Length function '{length_function_name}' not supported")
                raise NotImplementedError(
                    f"Error setting length function, neither python built-in nor valid tiktoken name passed. {ex.args}"
                )

    @staticmethod
    def get_language_model_name(llm: BaseLanguageModel) -> str:
        """Retrieve name of language model from object"""
        return llm.model_name

    @staticmethod
    def get_embedding_model_name(emb: Embeddings) -> str:
        """Retrieve name of embedding model name from object"""
        return emb.model

    def to_dict(self) -> dict:
        _data = self.dict()
        _data.pop("length_function", None)
        return _data

    @classmethod
    @abstractmethod
    def from_dict(cls, input_dict):
        pass
