from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel
from langchain.schema import BaseLanguageModel
from langchain.chat_models import ChatOpenAI


class LLMConfig(BaseModel):
    model: str
    temperature: Optional[float]
    openai_api_key: str

class BaseCheckResponse(BaseModel):
    input: BaseModel
    reasoning: str
    judgment: float

class BaseCheck(BaseModel, ABC):
    config: LLMConfig = None
    llm: BaseLanguageModel = None

    def __init__(self, **data):
        super().__init__(**data)  # for pydantic to not yell
        self.llm = ChatOpenAI(**data)

    @abstractmethod
    def check(cls, *args, **kwargs) -> BaseCheckResponse:
        """Checks the input and returns a response."""
