from typing import Dict, Any, Optional, List, Tuple, Union

import requests
from langchain.adapters.openai import convert_message_to_dict, convert_dict_to_message
from langchain.chains.base import logger
from langchain.utils import get_from_dict_or_env
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.utils import get_pydantic_field_names
from pydantic.v1 import Field, root_validator, BaseModel
from langchain.schema.messages import HumanMessage


class Perplexity:
    @classmethod
    def create(cls, model, messages, api_key):
        print(model, messages, api_key)
        url = "https://api.perplexity.ai/chat/completions"

        payload = {"model": model, "messages": messages}
        headers = {"accept": "application/json", "content-type": "application/json",
            "authorization": "Bearer " + api_key}

        response = requests.post(url, json=payload, headers=headers)

        print(response.text)
        return response.json()


class ChatPerplexity(BaseChatModel):
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"pplx_api_key": "PPLX_API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="pplx-7b-online", alias="model")
    """Model name to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    pplx_api_key: Optional[str] = Field(default=None, alias="api_key")

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended.""")
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                             f"Instead they were passed in as part of `model_kwargs` parameter.")

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["pplx_api_key"] = get_from_dict_or_env(values, "pplx_api_key", "PPLX_API_KEY")
        values['client'] = Perplexity
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {"model": self.model_name, **self.model_kwargs, }
        return params

    def completion_with_retry(self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = create_base_retry_decorator(error_types=[requests.ConnectionError], max_retries=1,
                                                      run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "pplx-chat"

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the openai client."""
        pplx_creds: Dict[str, Any] = {"model": self.model_name, "api_key": self.pplx_api_key}
        return {**self._default_params, **pplx_creds}

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        return combined

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any, ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, }
        response = self.completion_with_retry(messages=message_dicts, run_manager=run_manager, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message, generation_info=dict(finish_reason=res.get("finish_reason")), )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name, }
        return ChatResult(generations=generations, llm_output=llm_output)