
from .errors import ModelNotFound, ModelProviderNotFound, BadFormatForModelTag
from .mockai.info import MODELS as MOCKAI_MODELS
from .mockai.tokenizer import MockAITokenizer
from .openai import OpenAITokenizer
from .openai_info import OPEN_AI_MODELS


class Totokenizer:

    @classmethod
    def from_model(cls, model: str) -> OpenAITokenizer:
        try:
            provider, model_name = model.split("/", 1)
        except (ValueError, TypeError):
            raise BadFormatForModelTag(model)
        if provider == "openai":
            return OpenAITokenizer(model_name)  # type: ignore
        if provider == "mockai":
            return MockAITokenizer(model_name)  # type: ignore
        raise ModelProviderNotFound(provider)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError




class TotoModelInfo:

    @classmethod
    def from_model(cls, model: str):
        try:
            provider, model_name = model.split("/", 1)
        except ValueError:
            raise BadFormatForModelTag(model)
        if provider == "openai":
            if model_name not in OPEN_AI_MODELS:
                raise ModelNotFound(model_name)
            return OPEN_AI_MODELS[model_name]
        if provider == "mockai":
            if model_name not in MOCKAI_MODELS:
                raise ModelNotFound(model_name)
            return MOCKAI_MODELS[model_name]
        raise ModelProviderNotFound(provider)
