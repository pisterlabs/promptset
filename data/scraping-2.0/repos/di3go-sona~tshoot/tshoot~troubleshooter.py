"""Troubleshooter class to troubleshoot a problem with a given prompt and model"""
from typing import Optional

from openai import OpenAI

from .config import load_settings
from .prompts import PROMPTS
from .utils import _print_grey


class Troubleshooter:
    def __init__(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.settings = load_settings()
        self.prompt = PROMPTS[prompt or self.settings.PROMPT]
        self.model = model or self.settings.OPENAI_MODEL
        self.verbose = verbose or self.settings.VERBOSE

        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=self.settings.OPENAI_API_KEY,
        )
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.prompt,
            }
        ]

    def _add_new_message(self, role: str, content: str):
        """Add a new message to the chat"""
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def _add_message_part(self, content: str):
        """Add a new message part to the last message in the chat"""
        self.messages[-1]["content"] += content

    def _run(self):
        """Run the chat completion"""
        if self.verbose:
            _print_grey("[🤖->🌐]Running chat completion with prompt:")
            for message in self.messages:
                _print_grey(f"[{message['role']}] {message['content']}")

        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            stream=True,
        )

        self._add_new_message(role="assistant", content="")

        for chunk in chat_completion:
            try:
                part = chunk.choices[0].delta.content
                if part is not None:
                    self._add_message_part(part)
                    yield part
            except KeyboardInterrupt:
                # self._add_new_message(role="system", content="Chat interrupted")
                break

    def ask(self, question: str):
        """Ask a question to help troubleshoot the problem with the given prompt and model"""
        self.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )
        return self._run()
