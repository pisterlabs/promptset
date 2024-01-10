from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from google.cloud import firestore
from langchain.chat_models import ChatOpenAI

from config.loaders.firebase_loader import load_settings_from_firestore
from config.settings.api_settings import APISettings
from config.settings.celery_settings import CelerySettings
from config.settings.core_settings import CoreSettings
from config.settings.firebase_settings import FirebaseSettings
from config.settings.langsmith_settings import LangSmithSettings
from config.settings.proactive_messaging_settings import ProactiveMessagingSettings
from config.settings.user_identification_settings import UserIdentificationSettings

MAX_TOKENS = 1023


class AppConfig(ABC):
    """
    Application configuration manager.

    Manages configuration settings for the application using Pydantic models
    and loaders to obtain settings from various sources such as environment
    variables, Firestore, and INI files.
    """

    def __init__(self):
        """Initialize AppConfig with default settings."""
        self.api_settings = APISettings()
        self.core_settings = CoreSettings()
        self.firebase_settings = FirebaseSettings()
        self.langsmith_settings = LangSmithSettings()
        self.proactive_messaging_settings = ProactiveMessagingSettings()
        self.celery_settings = CelerySettings()
        self.user_identification_settings = UserIdentificationSettings()

    @property
    def vision_enabled(self) -> bool:
        return self.core_settings.vision_enabled

    @property
    def firebase_enabled(self) -> bool:
        return self.firebase_settings.enabled

    @property
    def langsmith_enabled(self) -> bool:
        """Determines if LangSmith feature is enabled."""
        return self.langsmith_settings.enabled

    @property
    def langsmith_api_key(self) -> str:
        """Retrieves the LangSmith API key."""
        api_key = self.langsmith_settings.api_key
        if api_key is None:
            raise ValueError("LangSmith API key is not set")
        return api_key

    @property
    def proactive_messaging_enabled(self) -> bool:
        return self.proactive_messaging_settings.enabled

    @property
    def proactive_message_interval_days(self) -> float:
        interval_days = self.proactive_messaging_settings.interval_days
        if interval_days is None:
            raise ValueError("Proactive messaging interval days is not set")
        return interval_days

    @property
    def proactive_system_prompt(self) -> str:
        system_prompt = self.proactive_messaging_settings.system_prompt
        if system_prompt is None:
            raise ValueError("Proactive system prompt is not set")
        return system_prompt

    @property
    def proactive_slack_channel(self) -> str:
        slack_channel = self.proactive_messaging_settings.slack_channel
        if slack_channel is None:
            raise ValueError("Proactive Slack channel is not set")
        return slack_channel

    @property
    def proactive_message_temperature(self) -> float:
        return self.proactive_messaging_settings.temperature

    def _validate_config(self) -> None:
        """Validate that required configuration variables are present."""
        assert (
            self.api_settings.openai_api_key
        ), "Missing configuration for openai_api_key"

        if self.langsmith_enabled:
            assert self.langsmith_api_key, "Missing configuration for LangSmith API key"

    def _apply_settings_from_companion(
        self, companion: firestore.DocumentSnapshot
    ) -> None:
        """
        Applies settings from the given companion Firestore document to the core settings
        of the application.

        Args:
            companion (firestore.DocumentSnapshot): Firestore document snapshot
                                                   containing companion settings.
        """
        settings_data = companion.to_dict() or {}

        # Special handling for 'prefix_messages_content' field
        if "prefix_messages_content" in settings_data:
            settings_data["prefix_messages_content"] = json.dumps(
                settings_data["prefix_messages_content"]
            )

        self.core_settings = CoreSettings(**settings_data)

    def _apply_proactive_messaging_settings_from_bot(
        self, bot_document: firestore.DocumentSnapshot
    ) -> None:
        """
        Applies proactive messaging settings from the provided bot document snapshot.

        This method extracts the proactive messaging settings from the Firestore
        document snapshot of the bot and applies them to the current configuration.
        It ensures that the proactive messaging feature and its related settings
        (interval days, system prompt, and Slack channel) are configured according
        to the bot's settings in Firestore.

        Args:
            bot_document (firestore.DocumentSnapshot): A snapshot of the Firestore
                                                      document for the bot.
        """
        self.proactive_messaging_settings = load_settings_from_firestore(
            ProactiveMessagingSettings, bot_document, "proactive_messaging"
        )
        logging.info("Proactive messaging settings applied from Firestore document.")

    def _apply_slack_tokens_from_bot(
        self, bot_document: firestore.DocumentSnapshot
    ) -> None:
        """
        Applies the Slack bot and app tokens from the provided bot document snapshot.

        Args:
            bot_document (firestore.DocumentSnapshot): A snapshot of the Firestore document for the bot.
        """
        slack_bot_token = bot_document.get("slack_bot_token")
        slack_app_token = bot_document.get("slack_app_token")

        # Update API settings with fetched tokens
        self.api_settings.slack_bot_token = slack_bot_token
        self.api_settings.slack_app_token = slack_app_token

    def _apply_user_identification_settings_from_bot(
        self, bot_document: firestore.DocumentSnapshot
    ) -> None:
        """
        Applies user identification settings from the provided bot document snapshot.

        Args:
            bot_document (firestore.DocumentSnapshot): A snapshot of the Firestore
                                                      document for the bot.
        """
        self.user_identification_settings = load_settings_from_firestore(
            UserIdentificationSettings, bot_document, "user_identification"
        )
        logging.info("User identification settings applied from Firestore document.")

    def _validate_and_apply_tokens(self):
        # Ensure that the tokens are not None before assignment
        if self.api_settings.slack_bot_token is not None:
            self.bot_token = self.api_settings.slack_bot_token
        else:
            raise ValueError("Slack bot token is missing in API settings.")

        if self.api_settings.slack_app_token is not None:
            self.app_token = self.api_settings.slack_app_token
        else:
            raise ValueError("Slack app token is missing in API settings.")

    def _apply_langsmith_settings(self):
        """
        Applies LangSmith settings if enabled.
        Sets LangSmith API key as an environment variable.
        """
        if self.langsmith_enabled:
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"

    @abstractmethod
    def load_config(self):
        """
        Abstract method to load configuration.

        This method should be implemented in derived classes to load configurations
        from specific sources.
        """

    def get_readable_config(self) -> str:
        """
        Retrieves a human-readable string of the current non-sensitive configuration.

        Returns:
            str: A string representing the current configuration excluding sensitive details.
        """
        return (
            f"Chat Model: {self.core_settings.chat_model}\n"
            f"System Prompt: {self.core_settings.system_prompt}\n"
            f"Temperature: {self.core_settings.temperature}\n"
            f"Frequency Penalty: {self.core_settings.frequency_penalty}\n"
            f"Vision Enabled: {'Yes' if self.vision_enabled else 'No'}"
        )


def init_chat_model(app_config: AppConfig) -> ChatOpenAI:
    """
    Initialize the langchain chat model.

    Args:
        app_config (AppConfig): Application configuration object.

    Returns:
        ChatOpenAI: Initialized chat model.
    """
    chat = ChatOpenAI(
        model=app_config.core_settings.chat_model,
        temperature=app_config.core_settings.temperature,
        model_kwargs={
            "frequency_penalty": app_config.core_settings.frequency_penalty,
        },
        openai_api_key=app_config.api_settings.openai_api_key,  # type: ignore
        openai_organization=app_config.api_settings.openai_organization,  # type: ignore
        max_tokens=MAX_TOKENS,
    )  # type: ignore
    return chat


def init_proactive_chat_model(app_config: AppConfig) -> ChatOpenAI:
    """
    Initializes a chat model specifically for proactive messaging.

    This function creates a chat model instance using settings configured for
    proactive messaging, including the temperature setting which influences the
    creativity of the generated messages.

    Args:
        app_config (AppConfig): The configuration object containing settings
                                     for proactive messaging.

    Returns:
        ChatOpenAI: An initialized chat model for proactive messaging.
    """
    proactive_temp = app_config.proactive_messaging_settings.temperature
    chat = ChatOpenAI(
        model=app_config.core_settings.chat_model,
        temperature=proactive_temp,
        openai_api_key=app_config.api_settings.openai_api_key,  # type: ignore
        openai_organization=app_config.api_settings.openai_organization,  # type: ignore
        max_tokens=MAX_TOKENS,
    )  # type: ignore
    return chat
