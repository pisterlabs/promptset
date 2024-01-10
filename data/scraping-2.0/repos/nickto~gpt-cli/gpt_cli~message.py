from __future__ import annotations

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from .model import OpenAiModel
from .role import Role
from .tokens import count_tokens


class Message(BaseModel):
    role: Role
    content: str | None
    model: OpenAiModel

    @computed_field
    @property
    def n_tokens(self) -> int:
        return self.count_tokens(self.content, self.model.name)

    @model_validator(mode="after")
    def content_can_be_none_only_for_system(self) -> Message:
        if self.content is None and self.role != Role.system:
            raise ValueError("If role is not 'system', content cannot be None.")
        return self

    @staticmethod
    def count_tokens(text: str | None, model: str) -> int:
        if text is None:
            return 0
        return count_tokens(text, model)

    def __str__(self):
        return f"{self.role.value.title()}: {self.content}"
