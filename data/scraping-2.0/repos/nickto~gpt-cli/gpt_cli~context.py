from __future__ import annotations

import io
import textwrap
from typing import Dict, List

import yaml

from .message import Message
from .model import OpenAiModel
from .role import Role


class Context:
    messages: List[Message]
    system: Message

    def __init__(self, model: OpenAiModel):
        self.model = model
        self.system = Message(role=Role.system, content=None, model=self.model)
        self.messages = []

    def add_message(self, message: Message) -> Context:
        self.messages.append(message)
        return self

    def save(self, filepath: str):
        messages = [
            message.model_dump(mode="json", exclude=["model", "n_tokens"])
            for message in self.messages
        ]
        if self.is_system_set():
            messages.insert(
                0,
                self.system.model_dump(mode="json", exclude=["model", "n_tokens"]),
            )

        # Instead of using pyyaml or ruamel.yaml, we'll just write the YAML file:
        # couldn't implement the formatting I liked with these libs
        with open(filepath, "w") as file:
            for message in messages:
                file.write(f"- role: {message['role']}\n")
                file.write(f"  content: >-\n")
                wrapped = textwrap.wrap(
                    message["content"],
                    width=80,
                    replace_whitespace=False,
                )
                indent = 4 * " "
                for _ in wrapped:
                    for line in _.split("\n"):
                        file.write(f"{indent}{line}\n")

    def load(self, filepath: str | io.TextIOWrapper):
        if isinstance(filepath, str):
            messages = yaml.safe_load(open(filepath, "r"))
        else:
            messages = yaml.safe_load(filepath)
        for m in messages:
            match m["role"]:
                case "system":
                    self.set_system(m["content"])
                case "user" | "assistant":
                    message = Message(
                        role=Role(m["role"]), content=m["content"], model=self.model
                    )
                    self.add_message(message)
                case _:
                    raise ValueError(f"Unknown role: {m['role']} in file {filepath}.")

        return self

    def is_system_set(self) -> bool:
        return self.system.content is not None

    def set_system(self, text: str) -> Context:
        self.system.content = text
        return self

    def _get_context(
        self,
        max_tokens: int = 2048,
        max_messages: int = 32 * 1024,  # just a very large number
    ) -> List[Message]:
        context_tokens = self.system.n_tokens
        context = []
        for message in reversed(self.messages):
            # Will token count be ok?
            ok_to_add = (context_tokens + message.n_tokens) <= max_tokens
            # Will message count be ok?
            ok_to_add = ok_to_add and (len(context) < max_messages)

            if ok_to_add:
                context.insert(0, message)
                context_tokens += message.n_tokens
            else:
                break

        if self.is_system_set():
            context.insert(0, self.system)

        return context

    @staticmethod
    def _context2dict(history: List[Message]) -> Dict[str, str]:
        messages = []
        for message in history:
            messages.append({"role": message.role.value, "content": message.content})

        return messages

    def get_messages(
        self,
        max_tokens: int = 2048,
        max_messages: int = 32 * 1024,  # just a very large number
    ) -> Dict[str, str]:
        context = self._get_context(max_tokens=max_tokens, max_messages=max_messages)
        return self._context2dict(context)
