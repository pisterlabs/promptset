from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytz
from langchain.schema import SystemMessage
from slack_sdk import WebClient
from slack_sdk.web.async_client import AsyncWebClient

from config.app_config import AppConfig, init_proactive_chat_model
from config.settings.proactive_messaging_settings import ProactiveMessagingSettings


def should_reschedule(
    old_settings: ProactiveMessagingSettings, new_settings: ProactiveMessagingSettings
) -> bool:
    """
    Determines if the interval in proactive messaging settings has changed.

    Args:
        old_settings (ProactiveMessagingSettings): The old settings.
        new_settings (ProactiveMessagingSettings): The new settings.

    Returns:
        bool: True if the interval settings have changed, False otherwise.
    """
    old_interval = old_settings.interval_days
    new_interval = new_settings.interval_days

    return old_interval != new_interval


def calculate_next_schedule_time(settings: ProactiveMessagingSettings) -> datetime:
    """
    Calculates the next schedule time for a proactive message based on the interval settings.

    Args:
        settings (ProactiveMessagingSettings): Configuration settings for proactive messaging.

    Returns:
        datetime: The calculated next schedule time in UTC for a proactive message.
    """
    interval_days = settings.interval_days
    if interval_days is None:
        raise ValueError("interval_days must be set for proactive messaging.")

    # Get current time in UTC
    now_utc = datetime.now(pytz.utc)

    # Calculate the next schedule time
    return now_utc + timedelta(days=interval_days * random.random() * 2)


async def generate_and_send_proactive_message_async(
    client: AsyncWebClient, app_config: AppConfig
) -> None:
    """
    Asynchronously generates a proactive message using the chat model based on
    the system prompt and sends the generated message to the specified Slack channel.

    Args:
        client (AsyncWebClient): The Slack client instance.
        app_config (AppConfig): The application configuration.
    """
    # Initialize chat model and generate message asynchronously
    chat = init_proactive_chat_model(app_config)
    system_prompt = SystemMessage(content=app_config.proactive_system_prompt)
    resp = await chat.agenerate([[system_prompt]])
    message = resp.generations[0][0].text

    # Send the generated message to the specified Slack channel
    channel = app_config.proactive_slack_channel
    await client.chat_postMessage(channel=channel, text=message)


def generate_and_send_proactive_message_sync(
    client: WebClient, app_config: AppConfig
) -> None:
    """
    Synchronously generates a proactive message using the chat model based on
    the system prompt and sends the generated message to the specified Slack channel.

    Args:
        client (WebClient): The Slack client instance.
        app_config (AppConfig): The application configuration.
    """
    # Initialize chat model and generate message synchronously
    chat = init_proactive_chat_model(app_config)
    system_prompt = SystemMessage(content=app_config.proactive_system_prompt)
    resp = chat.generate([[system_prompt]])  # Synchronous generation
    message = resp.generations[0][0].text

    # Send the generated message to the specified Slack channel
    channel = app_config.proactive_slack_channel
    client.chat_postMessage(channel=channel, text=message)
