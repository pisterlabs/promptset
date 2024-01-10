from typing import Dict, Tuple

from polly.client.cache import CacheClient
from polly.client.database import DatabaseClient
from polly.client.openai import OpenAIClient
from polly.client.gcloud import GoogleCloudClient
from polly.client.telegram import TelegramClient


class ClientContainer:

    def __init__(self,
                 telegram_token: str,
                 openai_token: str,
                 openai_model: Dict,
                 gcloud_cred: str,
                 database_url: str,
                 redis_cred: Tuple,
                 ):
        self._telegram_token = telegram_token
        self._openai_token = openai_token
        self._openai_model = openai_model
        self._database_url = database_url
        self._redis_host, self._redis_port, self._redis_pass = redis_cred
        self._gcloud_cred = gcloud_cred

        self.telegram_api = TelegramClient(token=self._telegram_token).__call__()
        self.openai_api = OpenAIClient(token=self._openai_token, model_dict=self._openai_model)
        self.gcloud_api = GoogleCloudClient(credential_file=self._gcloud_cred)
        self.database = DatabaseClient(database_url=self._database_url).__call__()
        self.cache = CacheClient(host=self._redis_host, port=self._redis_port, password=self._redis_pass)

    @classmethod
    def load(cls, config: Dict):
        """
        Load configuration from dictionary
        :param config:
        :return:
        """

        if config is None:
            raise ValueError('Config cannot be empty or null')

        telegram_token = config.get('TELEGRAM_TOKEN', '')
        if telegram_token == '':
            raise ValueError('Config key `TELEGRAM_TOKEN` cannot be empty.')

        openai_token = config.get('OPENAI_TOKEN', '')
        if openai_token == '':
            raise ValueError('Config key `OPENAI_TOKEN` cannot be empty.')

        gcloud_cred = config.get('GOOGLE_APPLICATION_CREDENTIALS', '')
        if gcloud_cred == '':
            raise ValueError('Config key `GOOGLE_APPLICATION_CREDENTIALS` cannot be empty.')

        database_url = config.get('DATABASE_URL', '')
        if database_url == '':
            raise ValueError('Config key `DATABASE_URL` cannot be empty.')

        redis_cred = (
            config.get('REDIS_HOST', ''),
            config.get('REDIS_PORT', ''),
            config.get('REDIS_PASSWORD', '')
        )

        openai_model = {
            'OPENAI_WHISPER': config.get('OPENAI_WHISPER', ''),
            'OPENAI_GPT': config.get('OPENAI_GPT')
        }

        return cls(
            telegram_token=telegram_token,
            openai_token=openai_token,
            openai_model=openai_model,
            gcloud_cred=gcloud_cred,
            database_url=database_url,
            redis_cred=redis_cred
        )
