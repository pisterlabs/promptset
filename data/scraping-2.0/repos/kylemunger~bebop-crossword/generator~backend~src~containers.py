from dependency_injector import containers, providers
from utils import Logger
from services.wiz import Wiz
from services.clue_generator import ClueGenerator
from openai import OpenAI

class Core(containers.DeclarativeContainer):
    config = providers.Configuration()
    config.log_level.from_env('LOG_LEVEL', default='INFO')
    config.openai_api_key.from_env('OPENAI_API_KEY')

    log = providers.Singleton(Logger, level=config.log_level)
    wiz = providers.Singleton(Wiz)
    openai_client = providers.Singleton(OpenAI, api_key=config.openai_api_key)
    clue_generator = providers.Singleton(ClueGenerator, client=openai_client)
