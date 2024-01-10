import requests
import logging
import os
import json
import tiktoken

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from functools import lru_cache
from langchain.llms import OpenAI, OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import completion_with_retry, update_token_usage
from langchain.schema import LLMResult

from utils.utils import dotdict


logger = logging.getLogger(__name__)


class GenerationModel(ABC):
    # used to generate text in general. e.g. could be using API, or local model
    @abstractmethod
    def generate(self, input_text, **gen_args):
        """
        Generate text from the model.
        """
        raise NotImplementedError

    def _cleaned_resp(self, data, prompt) -> "List[str]":
        # default helper function to clean extract the generated text from the returned json
        logger.debug("promopt:")
        logger.debug(prompt)
        cleaned_resps = []
        for gen_resp in data:
            logger.debug("raw response:")
            logger.debug(gen_resp['generated_text'])
            cleaned_resp = gen_resp['generated_text'].strip()
            if "\n" in cleaned_resp:
                cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
            logger.debug(f"cleaned response: {cleaned_resp}")
            cleaned_resps.append(cleaned_resp)
        return cleaned_resps


class HFAPIModel(GenerationModel):
    API_TOKEN = os.environ.get("HF_API_KEY")

    def __init__(self, model_name="gpt2-large", type="decoder"):
        self.type: str = type
        self.API_URL: str = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers: dict[str, str] = {"Authorization": f"Bearer {HFAPIModel.API_TOKEN}"}
        self.inference_args = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            'return_full_text': False,
        }
        # adjust inf args based on type
        if self.type == "encoder-decoder":
            self.inference_args.pop("return_full_text")
            self.inference_args["max_length"] = self.inference_args.pop("max_new_tokens")
        return

    def generate(self, input_text, **_args):
        data = {
            "inputs": input_text,
            "parameters": _args or self.inference_args
        }
        response = requests.post(self.API_URL, headers=self.headers, json=data)
        return response.json()


class TMPOpenAI(OpenAI):
    """Wrapper class for temporary OpenAI API access. Consistent with OpenAI class from langchain
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _to_openai_format(self, decoded_data: List[str]) -> Any:
        parsed_data = json.loads(decoded_data[0])
        out = dotdict(parsed_data)
        # clean up the exceess newlines
        for i in range(len(out.choices)):
            out.choices[i].text = out.choices[i].text.strip()
        return out

    def _format_request_data(self, **kwargs: Any) -> Any:
        prompt = kwargs.get('prompt', [''])
        prompt = [p.lstrip() for p in prompt]
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.7)
        max_tokens = kwargs.get('max_tokens', 64)
        n = kwargs.get('n', 1)
        stop = kwargs.get('stop', ['stop'])
        api_key = os.environ.get("TMP_OPENAI_API_KEY")

        return [prompt, temperature, top_p, max_tokens, n, api_key, stop]
    
    def _create(self, **kwargs: Any):
        gen_args = self._format_request_data(**kwargs)
        data = requests.post(
            url=os.environ.get("TMP_OPENAI_API") or '',
            json={
                'data': gen_args,
                'fn_index': 0
            }
        ).content
        decoded_data = json.loads(data.decode('utf-8'))['data']
        formatted_data = self._to_openai_format(decoded_data)
        return formatted_data

    def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self._create(**kwargs)

        return _completion_with_retry(**kwargs)
    

class OpenAIWrapper(OpenAI):
    def __init__(self, **kwargs):
        """wrapper class for OpenAI from langchain. Note that `import openai` CANNOT be executed before this class is instantiated.
        """
        model_name = kwargs.pop('model_name', 'code-davinci-002')
        if model_name in ['text003', 'chatgpt']:
            super().__init__(engine=model_name, **kwargs)
        else:
            super().__init__(model_name=model_name, **kwargs)
        return
    
    def truncate_sub_prompts(self, params, sub_prompts: List[List[str]]) -> List[List[str]]:
        """Truncate to leave room for max_tokens.
        """
        model_name = params.get('engine') or params.get('model')
        if 'code' in model_name:
            return sub_prompts
        # otherwise, truncate
        tokenizer = tiktoken.get_encoding('p50k_base')
        max_tokens = params["max_tokens"]
        model_max_tokens = self.modelname_to_contextsize(self.model_name)
        safe_zone = 20

        new_sub_prompts = []
        for sub_prompt in sub_prompts:
            new_subprompt = []
            for p in sub_prompt:
                encoded_p = tokenizer.encode(p)
                num_tokens = len(encoded_p)
                curr_total = num_tokens + max_tokens + safe_zone
                if curr_total > model_max_tokens:
                    # truncate
                    diff = curr_total - model_max_tokens
                    if diff > num_tokens / 2:
                        print(f"[WARNING] Truncating more than half of the subprompt with {num_tokens} tokens.")
                    new_p = tokenizer.decode(encoded_p[diff:])
                    new_subprompt.append(new_p)
                else:
                    new_subprompt.append(p)
            new_sub_prompts.append(new_subprompt)
        return new_sub_prompts
    
    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        import openai
        params = self._invocation_params
        if 'engine' in self.model_kwargs:
            openai.api_type = "azure"
            openai.api_base = os.environ.get('MS_OPENAI_API_BASE', '')
            openai.api_version = os.environ.get('MS_OPENAI_API_CHAT_VERSION')
            openai.api_key = os.environ.get('MS_OPENAI_API_KEY')
            params['engine'] = self.model_kwargs['engine']
            params['api_type'] = openai.api_type
            params['api_base'] = openai.api_base
            params['api_version'] = openai.api_version
            params['api_key'] = openai.api_key
        else:
            openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            openai.api_type = os.environ.get("OPENAI_API_TYPE", "open_ai")
            openai.api_version = None
            openai.api_key = os.environ.get('OPENAI_API_KEY')
        
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        sub_prompts = self.truncate_sub_prompts(params, sub_prompts)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}

        for _prompts in sub_prompts:
            # I am 100% not streaming
            # estimate total number of tokens
            response = completion_with_retry(self, prompt=_prompts, **params)
            choices.extend(response["choices"])
            # Can't update token usage if streaming
            update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(choices, prompts, token_usage)
    

class OpenAIChatWrapper(OpenAIChat):
    def __init__(self, prefix_sys_messages = '', **kwargs):
        """wrapper class for OpenAI from langchain. Note that `import openai` CANNOT be executed before this class is instantiated.
        """
        model_name = kwargs.pop('model_name', 'chatgpt')
        if model_name in ['chatgpt']:
            # azure
            super().__init__(engine=model_name, **kwargs)
        else:
            super().__init__(model_name=model_name, **kwargs)
        if prefix_sys_messages != '':
            self.prefix_messages = [{
                'role': 'system',
                'content': prefix_sys_messages
            }]
        return
    
    def _post_processing(self, llm_result: LLMResult) -> LLMResult:
        for gens in llm_result.generations:
            for gen in gens:
                gen_text: str = gen.text
                if gen_text.endswith('['):
                    gen_text = gen_text[:-1].strip()
                gen.text = gen_text
        return llm_result
    
    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        import openai
        if 'engine' in self.model_kwargs:
            openai.api_type = "azure"
            openai.api_base = os.environ.get('MS_OPENAI_API_BASE', '')
            openai.api_version = os.environ.get('MS_OPENAI_API_CHAT_VERSION')
            openai.api_key = os.environ.get('MS_OPENAI_API_KEY')
        else:
            openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            openai.api_type = os.environ.get("OPENAI_API_TYPE", "open_ai")
            openai.api_version = None
            openai.api_key = os.environ.get('OPENAI_API_KEY')
        llm_result = super()._generate(prompts, stop)
        llm_result = self._post_processing(llm_result)
        return llm_result