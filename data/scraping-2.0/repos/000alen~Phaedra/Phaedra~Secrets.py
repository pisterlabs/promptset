"""Secrets operations."""

import json
from typing import Dict

import openai


def get_secrets() -> Dict[str, str]:
    """Gets secrets from local file (secret.json).

    :return: Dictionary of secrets.
    :rtype: Dict[str, str]

    """

    with open("secrets.json") as file:
        secrets = json.load(file)
    return secrets


def get_secrets_remote() -> Dict[str, str]:
    """Gets secrets from remote file (secret.json) located in Google Drive. 
    Must be ran from Google Colaboratory.
    
    :return: Dictionary of secrets.
    :rtype: Dict[str, str]

    """

    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    with open("/content/drive/MyDrive/secrets.json") as file:
        secrets = json.load(file)
    return secrets


def set_secrets(secrets: Dict[str, str]):
    """Loads secrets (credentials).

    :param secrets: Dictionary of secrets.
    :type secrets: Dict[str, str]

    """

    openai.api_key = secrets["KEY"]
