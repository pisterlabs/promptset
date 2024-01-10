from typing import List, Optional, Type, Union
from enum import Enum, auto
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field, create_model
import openai, boto3, requests, json


class MessageRole(Enum):
    USER = auto()
    SYSTEM = auto()
    ASSISTANT = auto()


@dataclass
class Message:
    content: str = Field(default=None, repr=True)
    role: MessageRole = Field(default=MessageRole.USER, repr=False)
    name: Optional[str] = Field(default=None)

    def dict(self):
        assert self.content is not None, "Content must be set!"
        obj = {
            "role": self.role.name.lower(),
            "content": self.content,
        }
        if self.name and self.role == MessageRole.USER:
            obj["name"] = self.name
        return obj


@dataclass
class SystemMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.SYSTEM


@dataclass
class UserMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.USER


class PromptContext(BaseModel):
    prompt_string: str = Field(default="na")
    @property
    def return_string(self):
        return self.prompt_string

### ChatCompletionMessage
class ChatCompletionMessage (BaseModel):
    model: str = Field(default="gpt-3.5-turbo-0613")
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.1)
    functions: list[dict] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)


    @property
    def kwargs(self) -> dict:
        kwargs = {}
        kwargs["messages"] = self.messages
        kwargs["max_tokens"] = self.max_tokens
        kwargs["temperature"] = self.temperature
        kwargs["model"] = self.model
        print ("self.functions:", self.functions)
        kwargs["functions"] = self.functions
        return kwargs


### ChatCompletionMessage
class ChatCompletionMessageNoFunctions (BaseModel):
    model: str = Field(default="gpt-3.5-turbo-0613")
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.1)
    messages: list[dict] = Field(default_factory=list)
    

    @property
    def kwargs(self) -> dict:
        kwargs = {}
        kwargs["messages"] = self.messages
        kwargs["max_tokens"] = self.max_tokens
        kwargs["temperature"] = self.temperature
        kwargs["model"] = self.model
        return kwargs
