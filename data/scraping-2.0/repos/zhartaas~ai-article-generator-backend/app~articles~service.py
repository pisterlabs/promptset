from app.config import database
from .adapters.openai_service import LangChain
from .repository.repository import ArticleRepository
from pydantic import BaseSettings


class Config(BaseSettings):
    OPENAI_API_KEY: str


class Service:
    def __init__(self):
        config = Config()
        self.repository = ArticleRepository(database)
        self.llm = LangChain(config.OPENAI_API_KEY)


def get_service():
    svc = Service()
    return svc
