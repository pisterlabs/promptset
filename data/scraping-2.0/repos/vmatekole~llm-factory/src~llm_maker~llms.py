from injector import Module, provider, singleton
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms.gpt4all import GPT4All
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

from llm_maker.base_llm import Configuration, LLMModel

from . import logger


class OpenAIModel(LLMModel):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._config = config
        self._provider = OpenAI(
            model_name=config.model_name, temperature=config.temperature
        )


class GPT4AllModel(LLMModel):
    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._config: Configuration = config

        callbacks = [StreamingStdOutCallbackHandler()]
        self._provider: GPT4All = GPT4All(
            model=config.model_filepath, callbacks=callbacks, verbose=True
        )


class LLMFactory(Module):
    def __init__(self) -> None:
        super().__init__()

    models: dict[str, LLMModel] = {'OPENAI': OpenAIModel, 'GPT4ALL': GPT4AllModel}

    @provider
    @singleton
    def provide_model(self, config: Configuration) -> LLMModel:
        return LLMFactory.models[config.provider.upper()](config=config)
