import logging
import os

import openai
from openai import resources

from .completions import LibrettoCompletions, LibrettoChatCompletions
from .session import send_feedback_background
from .types import LibrettoConfig


logger = logging.getLogger(__name__)


class LibrettoChat(resources.Chat):
    completions: LibrettoChatCompletions

    def __init__(self, client: openai.Client, config: LibrettoConfig):
        super().__init__(client)
        self.completions = LibrettoChatCompletions(client, config)


class Client(openai.Client):
    config: LibrettoConfig
    completions: LibrettoCompletions
    chat: LibrettoChat

    def __init__(self, *args, libretto: LibrettoConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        config_dict = libretto._asdict() if libretto else {}
        if not config_dict.get("api_key"):
            config_dict["api_key"] = os.environ.get("LIBRETTO_API_KEY")
        self.config = LibrettoConfig(**config_dict)

        self.completions = LibrettoCompletions(self, self.config)
        self.chat = LibrettoChat(self, self.config)

    def send_feedback(
        self,
        *,
        feedback_key: str,
        api_key: str | None = None,
        better_response: str | None = None,
        rating: float | None = None,
    ):
        api_key = api_key or self.config.api_key
        if not api_key:
            logger.warning("Unable to send feedback to Libretto: missing api_key")
            return

        send_feedback_background(
            feedback_key=feedback_key,
            api_key=api_key,
            better_response=better_response,
            rating=rating,
        )
