import os
from loguru import logger
from rich import pretty
import openai

from .world import World
from .models.openai import ChatGPT


class App:
    name: str
    path: str
    world: World

    def __init__(self, name, path, world=None, db=None, llm=None, **kwargs):
        self.name = name
        self.path = path
        self.world = world or World(name, db=db, default_llm=llm)
    
    @classmethod
    def load(cls, path, db, llm, env=None):
        if llm is None:
            default_llm_id = os.environ.get('PXX_DEFAULT_LLM', 'chatgpt')

            if default_llm_id == 'chatgpt':
                api_key = os.environ.get('OPENAI_API_KEY')
                org_id = os.environ.get('OPENAI_ORG_ID')
                # TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization=org_id)'
                # openai.organization = org_id
                llm = llm or ChatGPT(id='default', api_key=openai.api_key, org_id=openai.organization)

        config = {
            'name': 'local',
            'path': path,
            'db': db,
            'llm': llm,
        }

        pretty.install()

        env = {**os.environ, **(env or {})}
        log_file_path = f"./log/{env.get('PXX_ENV', 'development')}.log"
        level = env.get('PXX_LOG_LEVEL', 'INFO')

        # Configure Loguru
        logger.remove()
        logger.add(
            log_file_path,
            rotation="10 MB",
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            backtrace=True
        )

        logger.info("Log file: " + log_file_path)
        logger.info("Log level: " + level)

        return cls(**config)
    
    def __repr__(self):
        return f'<App {self.name} path={self.path}>'

