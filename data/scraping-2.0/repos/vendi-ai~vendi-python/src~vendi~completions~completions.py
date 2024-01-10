from typing import List, Dict, Optional

import openai

from vendi.completions.schema import ChatCompletion
from vendi.core.http_client import HttpClient
from vendi.endpoints.schema import EndpointInfo
from vendi.models.schema import ModelProvider


class Completions:
    """
    Completions is the client to interact with the completions endpoint of the Vendi API.
    """

    def __init__(self, url: str, api_key: str):
        """
        Initialize the Completions client
        :param url: The URL of the Vendi API
        :param api_key: The API key to use for authentication
        """
        self.__api_key = api_key
        self.__client = HttpClient(
            url=url,
            api_key=api_key,
            api_prefix=f"/api/v1/providers/"
        )

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        frequency_penalty: Optional[float] = 0,
        presence_penalty: Optional[float] = 0,
        max_tokens: Optional[int] = 256,
        stop: Optional[List[str]] = None,
        n: Optional[int] = 1,
        top_p: Optional[float] = 1,
        top_k: Optional[int] = 40,
        temperature: Optional[float] = 0.7,
    ) -> ChatCompletion:
        """
        Create a completion on a language model with the given parameters
        :param model: The ID of the language model to use for the completion. Should be in the format of <provider>/<model_id>
        :param messages: The messages to use as the prompt for the completion
        :param frequency_penalty: The frequency penalty to use for the completion
        :param presence_penalty: The presence penalty to use for the completion
        :param max_tokens: The maximum number of tokens to generate for the completion
        :param stop: The stop condition to use for the completion
        :param n: The number of completions to generate
        :param top_p: The top p value to use for the completion
        :param top_k: The top k value to use for the completion
        :param temperature: The temperature value to use for the completion
        :return: The generated completion

        """
        data = {
            "messages": messages,
            "model": model,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "n": n,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }

        if stop is not None:
            data["stop"] = stop

        res = self.__client.post(
            uri=f"completions/",
            json_data=data
        )
        return res

    def create_batch(
        self,
        model,
        batch_messages: List[List[Dict[str, str]]],
        frequency_penalty: Optional[float] = 0,
        presence_penalty: Optional[float] = 0,
        max_tokens: Optional[int] = 256,
        stop: Optional[List[str]] = None,
        n: Optional[int] = 1,
        top_p: Optional[float] = 1,
        top_k: Optional[int] = 40,
        temperature: Optional[float] = 0.7,
    ) -> List[ChatCompletion]:
        """
        Create multiple completions on the same model with different prompts, while keeping the same parameters
        :param model: The ID of the language model to use for the completion. Should be in the format of <provider>/<model_id>
        :param batch_messages: A batch of multiple prompt messages to use for the completions
        :param frequency_penalty: The frequency penalty to use for the completion
        :param presence_penalty: The presence penalty to use for the completion
        :param max_tokens: The maximum number of tokens to generate for the completion
        :param stop: The stop condition to use for the completion
        :param n: The number of completions to generate
        :param top_p: The top p value to use for the completion
        :param top_k: The top k value to use for the completion
        :param temperature: The temperature value to use for the completion
        :return: The generated completions

        Examples:
        >>> from vendi import Vendi
        >>> client = Vendi(api_key="my-api-key")
        >>> completions = client.completions.create_batch(
        >>>     model="vendi/mistral-7b-instruct-v2",
        >>>     batch_messages=[
        >>>         [
        >>>             {
        >>>                 "role": "user",
        >>>                 "content": "Hello"
        >>>             }
        >>>     ],
        >>>     [
        >>>             {
        >>>                 "role": "user",
        >>>                 "content": "Hello what's up with you?"
        >>>            }
        >>>            ]
        >>>     ],
        >>> )


        """

        requests_body = [
            {
                "messages": message,
                "model": model,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "n": n,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
            }
            for message in batch_messages
        ]

        if stop is not None:
            for req in requests_body:
                req["stop"] = stop

        res = self.__client.post(
            uri=f"completions-many",
            json_data=
            {
                "requests": requests_body
            }
        )
        return res

    def create_many(
        self,
        models: List[str],
        messages: List[Dict[str, str]],
        frequency_penalty: Optional[float] = 0,
        presence_penalty: Optional[float] = 0,
        max_tokens: Optional[int] = 256,
        stop: Optional[List[str]] = None,
        n: Optional[int] = 1,
        top_p: Optional[float] = 1,
        top_k: Optional[int] = 40,
        temperature: Optional[float] = 0.7,
    ) -> List[ChatCompletion]:
        """
        Create multiple completions on different models with the same prompt and parameters
        :param models: A list of models to use for the completions. Each model should be in the format of <provider>/<model_id>
        :param messages: The messages to use as the prompt for the completions
        :param frequency_penalty: The frequency penalty to use for the completions
        :param presence_penalty: The presence penalty to use for the completions
        :param max_tokens: The maximum number of tokens to generate for the completions
        :param stop: The stop condition to use for the completions
        :param n: The number of completions to generate
        :param top_p: The top p value to use for the completions
        :param top_k: The top k value to use for the completions
        :param temperature: The temperature value to use for the completions
        :return: The generated completions

        Examples:
        >>> from vendi import Vendi
        >>> client = Vendi(api_key="my-api-key")
        >>> completions = client.completions.create_many(
        >>>     models=[
        >>>         "vendi/mistral-7b-instruct-v2",
        >>>         "openai/gpt-3.5-turbo",
        >>>         "openai/gpt4",
        >>>     ],
        >>>     messages=[
        >>>         {
        >>>             "role": "user",
        >>>             "content": "Hello"
        >>>         }
        >>>     ]
        >>> )
        """
        requests_body = [
            {
                "messages": messages,
                "model": model,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "n": n,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
            }
            for model in models
        ]

        if stop is not None:
            for req in requests_body:
                req["stop"] = stop

        res = self.__client.post(
            uri=f"completions-many/",
            json_data=
            {
                "requests": requests_body
            }
        )
        return res

    def available_endpoints(self, provider: ModelProvider) -> List[EndpointInfo]:
        """
        Get the list of available endpoints for the completions API
        :return: The list of available endpoints
        """
        res = self.__client.get(uri=f"{provider}/endpoints")
        return [EndpointInfo(**endpoint) for endpoint in res]
