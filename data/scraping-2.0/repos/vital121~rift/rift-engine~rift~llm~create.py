import functools
import weakref
from pydantic import BaseModel, SecretStr
from typing import Literal, Optional

from rift.llm.abstract import (
    AbstractCodeCompletionProvider,
    AbstractChatCompletionProvider,
)


class ModelConfig(BaseModel):
    chatModel: str
    completionsModel: str
    openaiKey: Optional[SecretStr] = None

    def __hash__(self):
        return hash((self.chatModel, self.completionsModel))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def create_chat(self) -> AbstractChatCompletionProvider:
        c = create_client(self.chatModel, self.openaiKey)
        assert isinstance(c, AbstractChatCompletionProvider)
        return c

    def create_completions(self) -> AbstractCodeCompletionProvider:
        return create_client(self.completionsModel, self.openaiKey)

    @classmethod
    def default(cls):
        return ModelConfig(
            completionsModel="gpt4all:ggml-replit-code-v1-3b",
            chatModel="gpt4all:ggml-mpt-7b-chat",
        )


CLIENTS = weakref.WeakValueDictionary()


def create_client(
    config: str, openai_api_key: Optional[SecretStr] = None
) -> AbstractCodeCompletionProvider:
    """Create a client for the given config. If the client has already been created, then it will return a cached one.

    Note that it uses a WeakValueDictionary, so if the client is no longer referenced, it will be garbage collected.
    This is useful because it means you can call create_client multiple times without allocating the same model, but
    if you need to dispose a model this won't keep a reference that prevents it from being garbage collected.
    """
    global CLIENTS

    if config in CLIENTS:
        return CLIENTS[config]
    else:
        client = create_client_core(config, openai_api_key)
        CLIENTS[config] = client
        return client


def create_client_core(
    config: str, openai_api_key: Optional[SecretStr]
) -> AbstractCodeCompletionProvider:
    assert ":" in config, f"Invalid config: {config}"
    type, rest = config.split(":", 1)
    type = type.strip()
    if "@" in rest:
        name, path = rest.split("@", 1)
    else:
        name = rest
        path = ""
    name = name.strip()
    path = path.strip()
    if type == "hf":
        from rift.llm.hf_client import HuggingFaceClient

        return HuggingFaceClient(name)
    elif type == "openai":
        from rift.llm.openai_client import OpenAIClient

        kwargs = {}
        if name:
            kwargs["default_model"] = name
        if openai_api_key:
            kwargs["api_key"] = openai_api_key
        if path:
            kwargs["api_url"] = path
        return OpenAIClient.parse_obj(kwargs)

    elif type == "gpt4all":
        from rift.llm.gpt4all_model import Gpt4AllSettings, Gpt4AllModel

        kwargs = {}
        if name:
            kwargs["model_name"] = name
        if path:
            kwargs["model_path"] = path
        settings = Gpt4AllSettings.parse_obj(kwargs)
        return Gpt4AllModel(settings)

    else:
        raise ValueError(f"Unknown model: {config}")
