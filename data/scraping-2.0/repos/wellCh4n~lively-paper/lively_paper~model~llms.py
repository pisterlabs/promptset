from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel


class LLMs(object):
    def __init__(self) -> None:
        super().__init__()


openai: BaseChatModel = ChatOpenAI(streaming=True)
