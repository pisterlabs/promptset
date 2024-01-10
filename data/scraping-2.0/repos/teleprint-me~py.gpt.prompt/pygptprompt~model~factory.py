"""
pygptprompt/model/factory.py
"""
from pygptprompt.config.manager import ConfigurationManager
from pygptprompt.model.base import ChatModel
from pygptprompt.model.llama_cpp import LlamaCppModel
from pygptprompt.model.openai import OpenAIModel


class ChatModelFactory:
    """
    A factory for creating chat model instances based on the provider.

    Attributes:
        config (ConfigurationManager): The configuration manager instance.
        provider_map (dict): A dictionary mapping provider keys to their corresponding classes.
    """

    def __init__(self, config: ConfigurationManager):
        """
        Initializes the ChatModelFactory with the given configuration manager.

        Args:
            config (ConfigurationManager): The configuration manager instance.
        """
        self.config = config
        self.provider_map = {
            "openai": OpenAIModel,
            "llama_cpp": LlamaCppModel,
        }

    def create_model(self, provider: str) -> ChatModel:
        """
        Creates and returns a chat model instance based on the provider.

        Args:
            provider (str): The provider key.

        Returns:
            ChatModel: The chat model instance.

        Raises:
            ValueError: If the provider is unknown.
        """
        provider_config = self.config.get_value(provider)

        if provider_config is None or "provider" not in provider_config:
            raise ValueError(f"Unknown provider: {provider}")

        provider_key = provider_config["provider"]

        if provider_key not in self.provider_map:
            raise ValueError(f"Unknown provider: {provider}")

        return self.provider_map[provider_key](self.config)
