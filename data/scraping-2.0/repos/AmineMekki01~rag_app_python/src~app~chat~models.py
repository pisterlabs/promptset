from pydantic import BaseModel, Field

from src.app.chat.constants import ChatRolesEnum, ModelsEnum
from src.app.core.models import TimestampAbstractModel
from src.app.chat.exceptions import OpenAIFailedProcessingException
from typing import Optional
from datetime import datetime


class BaseMessage(BaseModel):
    id: str = Field(default="")
    chat_id: str = Field(default="")
    model: ModelsEnum = Field(default=ModelsEnum.GPT4.value)
    userId: Optional[str] = None
    agent_role: str = Field(default=ChatRolesEnum.ASSISTANT.value)
    user_message: str = Field(default="")
    answer: str = Field(default="")
    augmented_message: str = Field(default="")


class Message(TimestampAbstractModel, BaseMessage):
    role: Optional[ChatRolesEnum] = None


class Message(BaseMessage):
    role: Optional[ChatRolesEnum] = None


class ChatSummary(BaseModel):
    id: str
    user_id: str
    title: str
    model: str
    agent_role: str
    created_at: datetime
    updated_at: datetime


class Chunk(BaseModel):
    id: str
    created: int = Field(default=0)
    model: ModelsEnum = Field(default="gpt-4-0613")
    content: str
    finish_reason: str | None = None

    @classmethod
    def from_chunk(cls, chunk):
        delta_content: str = cls.get_chunk_delta_content(chunk=chunk)
        return cls(
            id=chunk["id"],
            created=chunk["created"],
            model=chunk["model"],
            content=delta_content,
            finish_reason=chunk["choices"][0].get("finish_reason", None),
        )

    @staticmethod
    def get_chunk_delta_content(chunk: dict | str) -> str:
        try:
            match chunk:
                case str(chunk):
                    return chunk
                case dict(chunk):
                    return chunk["choices"][0]["delta"].get("content", "")
        except Exception:
            raise OpenAIFailedProcessingException
