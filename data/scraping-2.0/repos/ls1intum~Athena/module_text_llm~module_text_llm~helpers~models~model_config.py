from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain.base_language import BaseLanguageModel


class ModelConfig(BaseModel, ABC):
    
    @abstractmethod
    def get_model(self) -> BaseLanguageModel:
        pass
