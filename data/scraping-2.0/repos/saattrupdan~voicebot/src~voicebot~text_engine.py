"""The engine that produces new responses."""

from omegaconf import DictConfig
import openai
from dotenv import load_dotenv
import os
import logging
import datetime as dt


load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Du hedder Robert og er en dansk stemmerobot. Du er sød, rar og hjælpsom, og dine svar
er altid super korte og præcise.
"""


class TextEngine:
    """The engine that produces new responses.

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def reset_conversation(self) -> None:
        """Reset the conversation, only keeping the system prompt."""
        self.conversation = [dict(role="system", content=SYSTEM_PROMPT.strip())]

    def generate_response(
        self,
        prompt: str,
        last_response_time: dt.datetime,
        current_response_time: dt.datetime,
    ) -> str | None:
        """Generate a new response from a prompt.

        Args:
            prompt: Prompt to generate a response from.
            last_response_time: Time of the last response.
            current_response_time: Time of the current response.

        Returns:
            Generated response, or None if prompt should not be responded to.
        """
        response_delay = current_response_time - last_response_time
        seconds_since_last_response = response_delay.total_seconds()
        if seconds_since_last_response > self.cfg.follow_up_max_seconds:
            self.reset_conversation()

        self.conversation.append(dict(role="user", content=prompt))
        llm_answer = openai.ChatCompletion.create(
            model=self.cfg.text_model_id,
            messages=self.conversation,
            temperature=self.cfg.temperature,
        )
        response: str = llm_answer.choices[0].message.content.strip()
        self.conversation.append(dict(role="assistant", content=response))
        logger.info(f"Generated the response: {response!r}")
        return response
