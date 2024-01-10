from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from PromptKernel.LLMClient.SingletonMeta import SingletonMeta


class ChatOpenAIClient(SingletonMeta):
    model: BaseLanguageModel

    def __init__(self, model):
        self.model = ChatOpenAI(model=model, temperature=0)

    def get_model(self) -> BaseLanguageModel:
        return self.model
