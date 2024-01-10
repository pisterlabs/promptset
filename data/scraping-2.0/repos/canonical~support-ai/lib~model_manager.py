"""Language model manager (LLM) and factory for LLMs and embeddings"""
from abc import ABC, abstractmethod
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    LlamaCppEmbeddings,
    OpenAIEmbeddings,
)
from langchain.llms import HuggingFacePipeline, LlamaCpp, OpenAI
from lib.const import CONFIG_TYPE, CONFIG_LLM, CONFIG_EMBEDDINGS, \
        CONFIG_MODEL, CONFIG_HUGGINGFACE_PIPELINE, CONFIG_LLAMACPP, \
        CONFIG_OPENAI, CONFIG_LLM_OPENAI_API_KEY


class LLMFactory(ABC):
    @abstractmethod
    def create_llm(self):
        return NotImplemented

    @abstractmethod
    def create_embeddings(self):
        return NotImplemented


class HuggingFaceFactory(LLMFactory):
    def __init__(self, config):
        self.model = config[CONFIG_MODEL]
        if not self.model:
            raise ValueError("Missing model in llm config")

    def create_llm(self):
        return HuggingFacePipeline.from_model_id(model_id=self.model,
                                                 task='text-generation')

    def create_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.model)

class LlamaCppFactory(LLMFactory):
    def __init__(self, config):
        self.model = config[CONFIG_MODEL]
        if not self.model:
            raise ValueError("Missing model in llm config")

    def create_llm(self):
        return LlamaCpp(model_path=self.model, n_ctx=4096)

    def create_embeddings(self):
        return LlamaCppEmbeddings(model_path=self.model, n_ctx=4096)

class OpenAIFactory(LLMFactory):
    def __init__(self, llm_config):
        self.model = llm_config[CONFIG_MODEL]
        self.api_key = llm_config[CONFIG_LLM_OPENAI_API_KEY]
        if not self.api_key:
            raise ValueError("Missing api_key in llm config")

    def create_llm(self):
        return OpenAI(
            openai_api_key=self.api_key,
            model_name=self.model,
        )

    def create_embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model=self.model, openai_api_key=self.api_key)

_factories: dict = {
    CONFIG_HUGGINGFACE_PIPELINE: HuggingFaceFactory,
    CONFIG_LLAMACPP: LlamaCppFactory,
    CONFIG_OPENAI: OpenAIFactory
}

class ModelManager:
    def __init__(self, config) -> None:
        if CONFIG_LLM not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_LLM}')
        llm_config = config[CONFIG_LLM]
        if CONFIG_TYPE not in llm_config:
            raise ValueError(f'The llm configuration doesn\'t contain {CONFIG_TYPE}')
        if CONFIG_MODEL not in llm_config:
            raise ValueError(f'The llm configuration doesn\'t contain {CONFIG_MODEL}')
        llm_type = llm_config[CONFIG_TYPE]
        if llm_type not in _factories:
            raise ValueError(f'Unknown llm type: {llm_type}')
        self.llm = _factories[llm_type](llm_config).create_llm()

        if CONFIG_EMBEDDINGS not in config:
            self.embeddings = None
        else:
            embeddings_config = config[CONFIG_EMBEDDINGS]
            if CONFIG_TYPE not in embeddings_config:
                raise ValueError(f'The embeddings configuration doesn\'t contain {CONFIG_TYPE}')
            if CONFIG_MODEL not in embeddings_config:
                raise ValueError(f'The embeddings configuration doesn\'t contain {CONFIG_MODEL}')
            embeddings_type = embeddings_config[CONFIG_TYPE]
            if embeddings_type not in _factories:
                raise ValueError(f'Unknown embeddings type: {embeddings_type}')
            self.embeddings = _factories[embeddings_type](embeddings_config).create_embeddings()
