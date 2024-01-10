from abc import ABC, abstractmethod
from langchain.callbacks.base import CallbackManager


class AgentInterface(ABC):
    @property
    @abstractmethod
    def callback_manager(self) -> CallbackManager:
        """Returns a valid callback manager for use by the agent"""
        pass

    @abstractmethod
    def prompt(self) -> str:
        """Returns a valid prompt for use by the agent"""
        pass
