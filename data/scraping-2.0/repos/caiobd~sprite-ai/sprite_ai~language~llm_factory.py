import os
from langchain.llms.base import LLM
from langchain.llms.ollama import Ollama
from langchain.llms.openai import OpenAI
from langchain.llms.together import Together
from langchain.llms.llamacpp import LlamaCpp
from llama_cpp import suppress_stdout_stderr
import platformdirs

from sprite_ai.utils.download import download_file
from sprite_ai.constants import APP_NAME


class LLMFactory:
    def _get_model_location(self, model_name: str) -> str:
        user_data_location = platformdirs.user_data_path(
            appname=APP_NAME,
            appauthor=None,
            version=None,
            roaming=False,
            ensure_exists=True,
        )
        user_models_location = user_data_location / 'models'
        user_models_location.mkdir(exist_ok=True)

        model_location = user_models_location / f'{model_name}.gguf'
        model_location = str(model_location)
        return model_location

    def _build_llamacpp(
        self,
        model_name: str,
        url: str,
        context_size: int,
        temperature: float,
        stop_strings: list[str],
    ) -> LLM:
        model_location = self._get_model_location(model_name)
        if not os.path.isfile(model_location):
            download_file(url, model_location)
        with suppress_stdout_stderr():
            llm = LlamaCpp(
                model_path=model_location,
                n_ctx=context_size,
                # n_gpu_layers=40,
                temperature=temperature,
                echo=False,
                stop=stop_strings,
            )  # type: ignore
        return llm

    def build(
        self,
        model_name: str,
        context_size: int,
        temperature: float = 0.7,
        url: str = '',
        stop_strings: list[str] | None = None,
        api_key: str = '',
    ) -> LLM:
        if stop_strings is None:
            stop_strings = []

        try:
            prefix_end_position = model_name.index('/')
        except ValueError as e:
            raise ValueError('Missing model backend in model name', e)
        model_name_start_position = prefix_end_position + 1
        model_prefix = model_name[:prefix_end_position]
        model_name = model_name[model_name_start_position:]

        if model_prefix == 'ollama':
            llm = Ollama(
                model=model_name,
                num_ctx=context_size,
                temperature=temperature,
                base_url=url,
                stop=stop_strings,
            )
        elif model_prefix == 'openai':
            url = url if url else None
            llm = OpenAI(
                model=model_name,
                max_tokens=context_size,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=url,
            )
        elif model_prefix == 'together':
            url = url if url else 'https://api.together.xyz/inference'
            llm = Together(
                model=model_name,
                max_tokens=context_size,
                temperature=temperature,
                together_api_key=api_key,
                base_url=url,
            )
        elif model_prefix == 'local':
            llm = self._build_llamacpp(
                model_name, url, context_size, temperature, stop_strings
            )
        else:
            raise ValueError('Unsuported model type')

        return llm
