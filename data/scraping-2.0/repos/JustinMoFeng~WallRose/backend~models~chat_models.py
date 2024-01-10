from typing import Literal
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageToolCall as ToolCall

from . import PyObjectId

class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    name: str
    tool_call_id: str
    content: str

Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage

class Chat(BaseModel):
    id: PyObjectId | None = Field(alias="_id", default=None)
    owner: PyObjectId
    name: str | None = None
    messages: list[Message]