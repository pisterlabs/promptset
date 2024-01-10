from dependency_injector import containers, providers

from t_backend import routers
from t_backend.database import Database
from t_backend.repositories.chat import ChatRepository
from t_backend.repositories.message import MessageRepository
from t_backend.repositories.openai import OpenAIRepository
from t_backend.services.chat import ChatService
from t_backend.services.message import MessageService
from t_backend.settings import settings
from redis import Redis


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[routers])

    config = providers.Configuration()
    db = providers.Singleton(Database, db_url=settings.SQLALCHEMY_DATABASE_URL)
    redis_client = providers.Singleton(Redis, host=settings.REDIS_URL, port=6379)

    # repositories
    openai_repository = providers.Factory(OpenAIRepository, redis_client=redis_client)
    chat_repository = providers.Factory(
        ChatRepository, session_factory=db.provided.session
    )
    message_repository = providers.Factory(
        MessageRepository,
        session_factory=db.provided.session,
        redis_client=redis_client,
    )
    # services
    chat_service = providers.Factory(
        ChatService,
        chat_repository=chat_repository,
        message_repository=message_repository,
    )
    message_service = providers.Factory(
        MessageService,
        message_repository=message_repository,
        openai_repository=openai_repository,
    )
