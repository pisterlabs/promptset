"""Module for handling AI requests."""

from dataclasses import dataclass, field
import os
from openai import OpenAI


@dataclass
class AIHandler:
    """Class for handling AI requests."""

    settings_file_path: os.PathLike
    prompt_file_path: os.PathLike
    model: str = "gpt-4"
    settings: str = field(init=False)
    prompt: str = field(init=False)
    client: OpenAI = field(init=False)

    def __post_init__(self):
        """Initialize the AIHandler class."""
        self.settings = self.load_settings()
        self.prompt = self.load_prompt()
        self.client = OpenAI()

    def load_settings(self) -> str:
        """Load the settings from the settings file."""
        with open(self.settings_file_path, encoding="utf-8") as settings_file:
            return settings_file.read()

    def load_prompt(self) -> str:
        """Load the prompt from the prompt file."""
        with open(self.prompt_file_path, encoding="utf-8") as prompt_file:
            return prompt_file.read()

    def compose_message(self) -> list[dict[str, str]]:
        """Compose a message using the settings and prompt."""

        message = [
            {
                "role": "system",
                "content": self.settings,
            },
            {
                "role": "user",
                "content": self.prompt,
            },
        ]
        return message

    def get_completion(self) -> str:
        """Get a completion from the OpenAI API."""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.compose_message(),
        )
        return completion.choices[0].message.content
