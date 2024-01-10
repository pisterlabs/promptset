from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.load.serializable import Serializable
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from typing import Any, List, Mapping, Optional
import json
import requests


class _BaseCustomYandexGPT(Serializable):
    iam_token: str = ""
    api_key: str = ""
    x_folder_id: str = ""
    finetuned_model_id: str = ""
    model_name: str = "general"
    temperature: Optional[float] = 0.7
    max_tokens: int = 7400
    stop: Optional[List[str]] = None
    url: str = "https://llm.api.cloud.yandex.net/llm/v1alpha/instruct"

    @property
    def _llm_type(self) -> str:
        return "yandex_gpt"


class YandexCustomGPT(_BaseCustomYandexGPT, LLM):
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "finetuned_model_id": self.finetuned_model_id,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.finetuned_model_id or not self.api_key:
            headers = {"Authorization": f"Bearer {self.iam_token}", "x-folder-id": f"{self.x_folder_id}"}
        else:
            headers = {"Authorization": f"Api-Key {self.api_key}"}
            
        json_body = {
            "model": "general",
            "request_text": prompt,
            "generation_options": {
                "max_tokens": self.max_tokens, 
                "temperature": self.temperature
            }
        }
        
        if self.finetuned_model_id:
            json_body["instruction_uri"] = f"ds://{self.finetuned_model_id}"
            
        result = requests.post(
            url=self.url,
            headers=headers,
            json=json_body
        )
        text = json.loads(result.text)['result']['alternatives'][0]['text']
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
