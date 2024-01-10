from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import messages_to_dict, messages_from_dict, BaseChatMessageHistory
from langchain.chains import ConversationChain
from langchain.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import PostgresChatMessageHistory
from .chat_history import ChatHistory
from abc import ABCMeta, abstractclassmethod
import json
import os

class DataBaseChatHistory(ChatHistory, metaclass=ABCMeta):
    database_name: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str

    def __init__(
        self,
        session_id: str,
        tablename: str = 'message_store'
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.tablename = tablename

    @abstractclassmethod
    def _connect_db(self) -> str:
        pass

    @abstractclassmethod
    def load_history(self) -> BaseChatMessageHistory:
        pass

    @abstractclassmethod
    def add_chat_history(self, human: str = '', ai: str = ''):
        pass

    @abstractclassmethod
    def clean_history(self):
        pass

    @abstractclassmethod
    def messages_to_stringify_json(self, indent: int = 2, ascii: bool = False) -> str:
        pass


class PostgresChatHistory(DataBaseChatHistory):
    database_name: str = 'postgresql'

    def __init__(self, session_id: str, tablename: str = 'message_store') -> None:
        super().__init__(session_id, tablename)
        self.DB_USER = os.getenv('POSTGRES_USER')
        self.DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
        self.DB_HOST = os.getenv('POSTGRES_HOST')
        self.DB_PORT = os.getenv('POSTGRES_PORT')
        self._history = self.load_history()

    def _connect_db(self) -> str:
        return f'{self.database_name}://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}'

    def load_history(self) -> BaseChatMessageHistory:
        try:
            history = PostgresChatMessageHistory(session_id=self.session_id, connection_string=self._connect_db())
            return history

        except:
            raise

    def add_chat_history(self, human: str = '', ai: str = ''):
        self._history.add_user_message(human)
        self._history.add_ai_message(ai)

    def clean_history(self):
        if len(self._history.messages) < 1:
            return

        self._history.clear()

    def messages_to_stringify_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        messages = json.dumps(
            messages_to_dict(self._history.messages),
            indent=indent,
            ensure_ascii=ensure_ascii
        )

        return messages