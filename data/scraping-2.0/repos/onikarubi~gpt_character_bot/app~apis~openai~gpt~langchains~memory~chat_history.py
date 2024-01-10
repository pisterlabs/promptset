from abc import ABCMeta, abstractclassmethod
from langchain.schema import BaseChatMessageHistory, BaseMessage
from langchain.schema import AIMessage, HumanMessage

class ChatHistory(metaclass=ABCMeta):
    human_message: HumanMessage
    ai_message: AIMessage

    @abstractclassmethod
    def load_history(self) -> BaseChatMessageHistory:
        pass

    @abstractclassmethod
    def add_chat_history(self, human: str = '', ai: str = ''):
        pass

    @abstractclassmethod
    def clean_history(self):
        pass

