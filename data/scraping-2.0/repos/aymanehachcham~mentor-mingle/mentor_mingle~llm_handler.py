import logging
import os
from pathlib import Path
from typing import Any, Generator, Optional, Union

import openai
import redis
from dotenv import load_dotenv

from .config import Config
from .helpers.cache import Cache
from .persona.mentor import Mentor
from .utils import find_closest

load_dotenv()
logger = logging.getLogger(__name__)


class ChatHandler:
    """Handler for chat with GPT-3"""

    _memory: Cache = None

    def __init__(
        self,
        cache_client: Union[redis.Redis, Any] = None,
    ):
        """Initialize the chat handler"""
        openai.api_key = os.environ.get("OPENAI_KEY")
        self.agent = Mentor()

        # Load config
        self.model = Config.from_toml(Path(find_closest("config.toml"))).models.gpt3

        # Initialize memory
        if self._memory is None:
            self._memory = Cache(client=cache_client)

    @property
    def memory(self) -> Cache:
        """Get the memory"""
        return self._memory

    @property
    def last_chat(self) -> Optional[dict]:
        """Get the memory"""
        return self._memory.get_map("last_chat")

    @last_chat.setter
    def last_chat(self, value: dict) -> None:
        """Set the memory"""
        self._memory.set_map(
            "last_chat",
            {
                "query": value.get(b"query", ""),
                "assistant": value.get(b"assistant", ""),
            },
        )

    def stream_chat(self, user_prompt: str) -> Generator[str, None, None]:
        """
        Stream a chat with GPT-3

        Args:
            user_prompt (str): The user's prompt

        Returns:
            None
        """
        intro_session = [
            {"role": "system", "content": self.agent.persona},
            {
                "role": "user",
                "content": f"User: {user_prompt}",
            },
            {
                "role": "system",
                "content": self.agent.answer_format,
            },
        ]
        if self.last_chat:
            intro_session += [
                {
                    "role": "user",
                    "content": f"{self.last_chat.get(b'query', '')}",
                }
            ]
        completion = openai.ChatCompletion.create(
            model=self.model.name,
            messages=intro_session,
            **self.model.config.model_dump(),
        )
        for chunk in completion:
            content = chunk.choices[0].delta.get("content", "")
            if content != "":
                yield content
