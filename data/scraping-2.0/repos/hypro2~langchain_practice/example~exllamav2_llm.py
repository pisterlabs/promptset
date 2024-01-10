from abc import ABC
import torch
import sys
import random
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Mapping

from pydantic import Field
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.callbacks.manager import CallbackManagerForLLMRun

from exllamav2 import ExLlamaV2,ExLlamaV2Config,ExLlamaV2Cache,ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler, ExLlamaV2StreamingGenerator

class ExllamaLLM(LLM, ABC):
    model_folder_path: str = Field(None, alias='model_folder_path')
    model_name: str = Field(None, alias='model_name')
    backend: Optional[str] = 'llama'
    temperature: Optional[float] = 0.01
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 1024
    repetition_penalty: Optional[float] = 1.15
    model: Any = None
    tokenizer: Any = None

    def __init__(self, model_folder_path, callbacks=None, **kwargs):
        super(ExllamaLLM, self).__init__()
        self.model_folder_path: str = model_folder_path
        self.callbacks = callbacks

    @property
    def _get_model_default_parameters(self):
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': self.model_name,
            'model_path': self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }

    @property
    def _llm_type(self) -> str:
        return 'llama'

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:

        params = {
            **self._get_model_default_parameters,
            **kwargs
        }

        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        config = ExLlamaV2Config()
        config.model_dir = self.model_folder_path
        config.max_seq_len = 4096
        config.prepare()

        model = ExLlamaV2(config)
        print(f"load model : {config.model_dir}")
        model.load()

        tokenizer = ExLlamaV2Tokenizer(config)
        cache = ExLlamaV2Cache(model)

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = params['temperature']
        settings.token_repetition_penalty = params['repetition_penalty']

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        generator.warmup()
        response = generator.generate_simple(prompt, settings, params['max_tokens'], seed=1234)

        if stop:
            response = enforce_stop_tokens(response, stop)

        model.unload()

        return response

    def stream(self,
               prompt: str,
               stop: Optional[List[str]] = None,
               run_manager: Optional[CallbackManagerForLLMRun] = None,
               **kwargs) -> Generator[Dict, None, None]:

        params = {
            **self._get_model_default_parameters,
            **kwargs
        }

        config = ExLlamaV2Config()
        config.model_dir = self.model_folder_path
        config.max_seq_len = 4096
        config.prepare()

        model = ExLlamaV2(config)
        print(f"load model : {config.model_dir}")
        model.load()

        tokenizer = ExLlamaV2Tokenizer(config)
        cache = ExLlamaV2Cache(model)

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = params['temperature']
        settings.token_repetition_penalty = params['repetition_penalty']

        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
        generator.warmup()
        sys.stdout.flush()

        input_ids = tokenizer.encode(prompt.text)
        generator.begin_stream(input_ids, settings)

        output = ""
        random.seed(1234)
        generated_tokens = 0

        while True:
            chunk, eos, token = generator.stream()
            output = output + chunk
            generated_tokens += 1
            yield chunk
            sys.stdout.flush()
            if eos or generated_tokens == params['max_tokens']:
                break

        try:
            model.unload()
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(str(e))
