from app.config import database
from .adapters.gpt import OpenAIService
from pydantic import BaseSettings
from .repository.repository import MessageRepository


class Config(BaseSettings):
    OPENAI_API_KEY: str


class Service:
    def __init__(
        self,
        repository: MessageRepository,
    ):
        config = Config()
        self.repository = repository
        self.openai_service = OpenAIService(config.OPENAI_API_KEY)


def get_service():
    repository = MessageRepository(database)
    svc = Service(repository)
    return svc
