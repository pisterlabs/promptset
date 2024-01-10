import logging

from .anthropic import AnthropicProvider
from .azure import AzureProvider
from .gemini import GeminiProvider
from .llava import LlavaProvider
from .openai import OpenAIProvider
from .provider import BaseProvider
from ..types import ModelProvider


class ProviderManager:
    def __init__(self, openai_provider: OpenAIProvider,
                 azure_provider: AzureProvider,
                 anthropic_provider: AnthropicProvider,
                 llava_provider: LlavaProvider,
                 gemini_provider: GeminiProvider):
        self._providers = {
            ModelProvider.OPENAI: openai_provider,
            ModelProvider.AZURE: azure_provider,
            ModelProvider.GEMINI: gemini_provider,
            ModelProvider.ANTHROPIC: anthropic_provider,
            ModelProvider.LLAVA: llava_provider
        }
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_provider(self, provider_type: ModelProvider) -> BaseProvider:
        return self._providers[provider_type]

