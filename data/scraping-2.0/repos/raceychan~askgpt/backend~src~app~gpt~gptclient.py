import abc
import functools
import typing as ty
from collections import deque
from contextlib import asynccontextmanager

import httpx
import openai
from openai.types import beta as openai_beta
from openai.types import chat as openai_chat
from src.adapters import cache
from src.app.gpt import model, params

MAX_RETRIES: int = 3


class ClientNotRegisteredError(Exception):
    def __init__(self, name: str):
        msg = f"Client {name} not registered"
        super().__init__(msg)


class APIKeyNotAvailableError(Exception):
    def __init__(self, api_type: str):
        msg = f"No API keys available for {api_type=}"
        super().__init__(msg)


class ClientRegistry:
    _registry: ty.ClassVar[dict[str, type["AIClient"]]] = {}

    @classmethod
    def register(cls, name: str):
        def inner[T: type["AIClient"]](client_cls: T) -> T:
            cls._registry[name] = client_cls
            return client_cls

        return inner

    def __getitem__(self, name: str) -> type["AIClient"]:
        return self._registry[name]


class AIClient(abc.ABC):
    async def complete(
        self,
        messages: list[model.ChatMessage],
        model: model.CompletionModels,
        user: str,
        stream: bool = True,
        **options: ty.Unpack[params.CompletionOptions],  # type: ignore
    ) -> ty.AsyncIterable[openai_chat.ChatCompletionChunk]:
        ...

    @classmethod
    @abc.abstractmethod
    def from_apikey(cls, api_key: str) -> ty.Self:
        ...


@ClientRegistry.register("openai")
class OpenAIClient(AIClient):
    def __init__(self, client: openai.AsyncOpenAI):
        self._client = client

    async def assistant(
        self,
        model: str,
        name: str,
        instructions: str,
        tools: list[openai_beta.assistant_create_params.Tool],
    ) -> openai_beta.Assistant:
        return await self._client.beta.assistants.create(
            model=model, name=name, instructions=instructions, tools=tools
        )

    async def create_thread(self) -> openai_beta.Thread:
        """
        https://platform.openai.com/docs/assistants/overview
        """
        return await self._client.beta.threads.create()

    async def complete(
        self,
        messages: list[model.ChatMessage],
        model: model.CompletionModels,
        user: str,
        stream: bool = True,
        **options: ty.Unpack[params.CompletionOptions],  # type: ignore
    ) -> ty.AsyncIterable[openai_chat.ChatCompletionChunk]:
        msgs = self.message_adapter(messages)
        resp = await self._client.chat.completions.create(
            messages=msgs,  # type: ignore
            model=model,
            stream=stream,
            user=user,
            **options,
        )

        return resp

    def message_adapter(
        self, messages: list[model.ChatMessage]
    ) -> list[dict[str, ty.Any]]:
        return [message.asdict(exclude={"user_id"}) for message in messages]

    @classmethod
    @functools.cache
    def from_apikey(cls, api_key: str) -> ty.Self:
        client = openai.AsyncOpenAI(api_key=api_key)
        return cls(client=client)

    @classmethod
    def build(
        cls,
        api_key: str,
        *,
        organization: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = MAX_RETRIES,
        default_headers: ty.Mapping[str, str] | None = None,
        default_query: ty.Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
        _strict_response_validation: bool = False,
    ) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )


# https://medium.com/@colemanhindes/unofficial-gpt-3-developer-faq-fcb770710f42
# Only 2 concurrent requests can be made per API key at a time.
class APIPool:
    def __init__(
        self,
        pool_key: cache.KeySpace,
        api_type: str,
        api_keys: ty.Sequence[str],
        cache: cache.Cache[str, str],
        client_registry: ClientRegistry = ClientRegistry(),
    ):
        self.pool_key = pool_key
        self.api_type = api_type
        self.api_keys = deque(api_keys)
        self._cache = cache
        self._client_registry = client_registry
        self._client_cache: dict[str, AIClient] = {}
        self.__started: bool = False

    @property
    def is_started(self) -> bool:
        return self.__started

    @property
    def client_factory(self):
        try:
            return self._client_registry[self.api_type]
        except KeyError:
            raise ClientNotRegisteredError(self.api_type)

    async def acquire(self):
        # Pop an API key from the front of the deque
        if not self.__started:
            raise Exception("APIPool not started")
        api_key = await self._cache.lpop(self.pool_key.key)
        if not api_key:
            raise Exception("No API keys available")
        return api_key

    async def release(self, api_key: str):
        # Push the API key back to the end of the deque
        await self._cache.rpush(self.pool_key.key, api_key)

    async def start(self):
        if not self.api_keys:
            raise APIKeyNotAvailableError(self.api_type)
        await self._cache.rpush(self.pool_key.key, *self.api_keys)
        self.__started = True

    async def close(self):
        # remove api keys from redis, clear client cache
        await self._cache.remove(self.pool_key.key)
        self.__started = False

    @asynccontextmanager
    async def reserve_client(self):
        api_key = await self.acquire()

        try:
            if client := self._client_cache.get(api_key):
                yield client
            else:
                client = self.client_factory.from_apikey(api_key)
                self._client_cache[api_key] = client
            yield client
        finally:
            await self.release(api_key)

    @asynccontextmanager
    async def lifespan(self):
        try:
            await self.start()
            yield self
        finally:
            await self.close()
