from getpass import getpass
from types import ModuleType
from typing import Optional

import openai

from jupyter_aichat.output import output, TemplateLoader

keyring: Optional[ModuleType]
try:
    import keyring
except ImportError:
    keyring = None


def authenticate() -> None:
    if openai.api_key:
        return
    if keyring is not None:
        password = keyring.get_password("api.openai.com", "jupyter-aichat")
    else:
        password = None
    if not password:
        output(TemplateLoader()["authentication_note"])
        password = getpass("Enter your OpenAI API key:")
        output(TemplateLoader()["keyring_tip"])
    openai.api_key = password


def save_api_key() -> None:
    if keyring:
        keyring.set_password("api.openai.com", "jupyter-aichat", openai.api_key)
    else:
        output(TemplateLoader()["keyring_not_installed"])
