from pathlib import Path
from typing import Dict, Generator, List

from langchain.chat_models import GigaChat
from langchain.schema import ChatMessage

from .cache import Cache
from .config import cfg

CACHE_LENGTH = int(cfg.get("CACHE_LENGTH"))
CACHE_PATH = Path(cfg.get("CACHE_PATH"))
REQUEST_TIMEOUT = int(cfg.get("REQUEST_TIMEOUT"))
DISABLE_STREAMING = str(cfg.get("DISABLE_STREAMING"))
DEFAULT_MODEL = str(cfg.get("DEFAULT_MODEL"))


class GigaChatClient:
    cache = Cache(CACHE_LENGTH, CACHE_PATH)

    def __init__(self, api_host: str, username: str, password: str) -> None:
        self.api_host = api_host
        self.giga = GigaChat(
            user=username,
            password=password,
            verify_ssl_certs=False,
            base_url=api_host,
            model=DEFAULT_MODEL,
        )

    @cache
    def _request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1,
        top_probability: float = 1,
    ) -> Generator[str, None, None]:
        """
        Make request to GigaChat via GigaChain SDK

        :param messages: List of messages {"role": user or assistant, "content": message_string}
        :param temperature: Float in 0.0 - 2.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :return: Response body JSON.
        """
        stream = DISABLE_STREAMING == "false"

        self.giga.temperature = temperature
        # top_probability

        payload = [ChatMessage(**message) for message in messages]

        if not stream:
            response = self.giga(payload)
            yield response.content
        else:
            for chunk in self.giga.stream(payload):
                yield chunk.content

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1,
        top_probability: float = 1,
    ) -> Generator[str, None, None]:
        """
        Generates single completion for prompt (message).

        :param messages: List of dict with messages and roles.
        :param temperature: Float in 0.0 - 1.0 range.
        :param top_probability: Float in 0.0 - 1.0 range.
        :param caching: Boolean value to enable/disable caching.
        :return: String generated completion.
        """
        yield from self._request(
            messages,
            temperature,
            top_probability
        )
