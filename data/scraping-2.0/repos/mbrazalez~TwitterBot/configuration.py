import os
import re
import openai


class BaseConfiguration(object):
    
    # AWS CONFIGURATION
    # > AWS SERVICE: ENVIRONMENT
    AWS_STAGE = os.getenv('AWS_STAGE', None)
    AWS_REGION = os.environ.get('AWS_REGION', None)
    AWS_ACCOUNT = os.environ.get('AWS_ACCOUNT', None)

    # > AWS SERVICE: S3
    S3_BUCKET_STORAGE = os.environ.get('S3_BUCKET_STORAGE', None)

    # > EXTERNAL SERVICES

    # CREDENTIALS FOR AUTH IN TWITTER API
    API_KEY = os.getenv('API_KEY', None)
    API_SECRET_KEY = os.getenv('API_SECRET_KEY', None)
    BEARER_TOKEN = os.getenv('BEARER_TOKEN', None)
    ACCESS_TOKEN = os.getenv('ACCESS_TOKEN', None)
    ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET', None)

    # KEY FOR AUTH IN OPENAI API
    openai.api_key = os.getenv('OPENAI_API_KEY', None)

    def __getattribute__(self, name):
        item = object.__getattribute__(self, name)
        return item

    @classmethod
    def get(cls, key):
        item = getattr(cls, key)
        if type(item) is str:
            item = cls.format(item)
        return item

    @classmethod
    def format(cls, _str):
        replace_keys = re.findall('{(.*?)}', _str)
        replace_items = {k: cls.get(k) for k in replace_keys}
        return _str.format(**replace_items)


configuration = BaseConfiguration()
