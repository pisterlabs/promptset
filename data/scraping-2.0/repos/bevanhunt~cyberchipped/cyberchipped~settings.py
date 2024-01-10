import os
from typing import TYPE_CHECKING, Any, Optional, Union

from dotenv import load_dotenv

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from openai import AsyncClient, Client
    from openai.types.chat import ChatCompletion

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Preserve SecretStr type when setting values."""
        field = self.model_fields.get(name)
        if field:
            annotation = field.annotation
            base_types = (
                annotation.__args__
                if getattr(annotation, "__origin__", None) is Union
                else (annotation,)
            )
            if SecretStr in base_types and not isinstance(value, SecretStr):
                value = SecretStr(value)
        super().__setattr__(name, value)


class ModelSettings(Settings):
    model: str

    @property
    def encoder(self):
        import tiktoken

        return tiktoken.encoding_for_model(self.model).encode


class ChatCompletionSettings(ModelSettings):
    model: str = Field(
        default="gpt-3.5-turbo-1106",
        description="The default chat model to use.",
    )

    async def acreate(self, **kwargs: Any) -> "ChatCompletion":
        from cyberchipped.settings import settings

        return await settings.openai.async_client.chat.completions.create(
            model=self.model, **kwargs
        )

    def create(self, **kwargs: Any) -> "ChatCompletion":
        from cyberchipped.settings import settings

        return settings.openai.client.chat.completions.create(
            model=self.model, **kwargs
        )


class AssistantSettings(ModelSettings):
    model: str = Field(
        default="gpt-3.5-turbo-1106",
        description="The default assistant model to use.",
    )


class ChatSettings(Settings):
    completions: ChatCompletionSettings = Field(default_factory=ChatCompletionSettings)


class OpenAISettings(Settings):
    model_config = SettingsConfigDict(env_prefix="_openai_")

    api_key: Optional[SecretStr] = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="Your OpenAI API key.",
    )

    organization: Optional[str] = Field(
        default=None,
        description="Your OpenAI organization ID.",
    )

    chat: ChatSettings = Field(default_factory=ChatSettings)
    assistants: AssistantSettings = Field(default_factory=AssistantSettings)

    @property
    def async_client(
        self, api_key: Optional[str] = None, **kwargs: Any
    ) -> "AsyncClient":
        from openai import AsyncClient

        if not (api_key or self.api_key):
            raise ValueError("No API key provided.")
        elif not api_key and self.api_key:
            api_key = self.api_key.get_secret_value()

        return AsyncClient(
            api_key=api_key,
            organization=self.organization,
            **kwargs,
        )

    @property
    def client(self, api_key: Optional[str] = None, **kwargs: Any) -> "Client":
        from openai import Client

        if not (api_key or self.api_key):
            raise ValueError("No API key provided.")
        elif not api_key and self.api_key:
            api_key = self.api_key.get_secret_value()

        return Client(
            api_key=api_key,
            organization=self.organization,
            **kwargs,
        )


class Settings(Settings):
    model_config = SettingsConfigDict(env_prefix="_")

    openai: OpenAISettings = Field(default_factory=OpenAISettings)

    log_level: str = Field(
        default="DEBUG",
        description="The log level to use.",
    )


settings = Settings()
