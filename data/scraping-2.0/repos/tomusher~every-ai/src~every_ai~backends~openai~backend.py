from dataclasses import dataclass
from typing import List, Literal, Optional

from every_ai import AIBackend, registry
from every_ai.exceptions import InvalidBackendConfigurationError

# Types
ChatModels = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
]

EmbeddingModels = Literal["text-embedding-ada-002"]


@dataclass
class ChatModelConfig:
    token_limit: int


@dataclass
class EmbeddingModelConfig:
    token_limit: int
    output_dimensions: int


@dataclass
class BackendConfig:
    api_key: str = ""
    chat_model: ChatModels = "gpt-3.5-turbo"
    embedding_model: EmbeddingModels = "text-embedding-ada-002"


# Constants

CHAT_MODELS = {
    "gpt-3.5-turbo": ChatModelConfig(token_limit=4096),
    "gpt-3.5-turbo-16k": ChatModelConfig(token_limit=16385),
    "gpt-4": ChatModelConfig(token_limit=8192),
    "gpt-4-32k": ChatModelConfig(token_limit=32768),
}

EMBEDDING_MODELS = {
    "text-embedding-ada-002": EmbeddingModelConfig(
        token_limit=8191, output_dimensions=1536
    )
}

# Implementation


@registry.register("openai")
class OpenAIBackend(AIBackend):
    config_class = BackendConfig

    def __init__(self, config: dict):
        try:
            from openai import OpenAI

            self.config = self.config_class(**config)
            api_key = self.config.api_key
            self.client = OpenAI(api_key=api_key)
        except AttributeError as e:
            raise InvalidBackendConfigurationError(
                "The api_key setting must be configured to use OpenAI"
            ) from e

    def chat(
        self, *, system_messages: Optional[List[str]] = None, user_messages: List[str]
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=[
                {"role": "system", "content": message}
                for message in system_messages or []
            ]
            + [{"role": "user", "content": message} for message in user_messages],
        )
        return completion.choices[0].message.content or ""

    def embed(self, inputs: List[str]) -> List[List[float]]:
        embeddings = self.client.embeddings.create(
            model=self.config.embedding_model, input=inputs
        )
        return [embedding.embedding for embedding in embeddings.data]

    @property
    def embedding_output_dimensions(self) -> int:
        model_metadata = EMBEDDING_MODELS[self.config.embedding_model]
        return model_metadata.output_dimensions
