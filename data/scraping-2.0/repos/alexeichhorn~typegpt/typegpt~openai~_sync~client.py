from typing import Mapping

import httpx
from openai import AzureOpenAI, OpenAI, resources
from openai._constants import DEFAULT_MAX_RETRIES
from openai._types import NOT_GIVEN, NotGiven
from openai.lib.azure import AzureADTokenProvider

from .chat_completion import TypeChatCompletion


class TypeChat(resources.chat.Chat):
    completions: TypeChatCompletion

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.completions = TypeChatCompletion(client)


class TypeOpenAI(OpenAI):
    chat: TypeChat

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client. See the [httpx documentation](https://www.python-httpx.org/api/#asyncclient) for more details.
        http_client: httpx.Client | None = None,
        # only needed to have same subclass capabilities (i.e. for Azure)
        _strict_response_validation: bool = False,
    ) -> None:
        super().__init__(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )
        self.chat = TypeChat(self)


class TypeAzureOpenAI(AzureOpenAI, TypeOpenAI):
    ...
